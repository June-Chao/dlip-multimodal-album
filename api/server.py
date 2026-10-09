"""Local album application. Library features never load ML dependencies."""
from __future__ import annotations

from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.library import Library, MAX_FILE_BYTES, inspect_image
from app.search_service import keyword_rank, fuse_rankings
from config import BASE_DIR, DATA_DIR, MODEL_DIR

Orientation = Literal["landscape", "portrait", "square"]


def call_ai(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(503, "AI 功能暂不可用，请检查本地模型文件、依赖和索引后重试。") from exc


class PhotoUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    favorite: bool | None = None
    caption: str | None = Field(default=None, max_length=2000)
    tags: list[str] | None = Field(default=None, max_length=30)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags):
        if tags is not None and any(len(tag) > 60 for tag in tags):
            raise ValueError("每个标签最多 60 字")
        return tags


class AlbumCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)

    @field_validator("name")
    @classmethod
    def nonblank(cls, name):
        if not name.strip():
            raise ValueError("请输入相册名称")
        return name.strip()


class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    k: int = Field(default=60, ge=1, le=200)
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    favorite: bool = False
    album_id: str | None = None
    orientation: Orientation | None = None

    @field_validator("q")
    @classmethod
    def nonblank(cls, q):
        if not q.strip():
            raise ValueError("请输入搜索内容")
        return q.strip()


class SimilarRequest(BaseModel):
    k: int = Field(default=30, ge=1, le=100)


class IndexRequest(BaseModel):
    languages: list[Literal["en", "zh"]] = Field(default=["en", "zh"], min_length=1, max_length=2)
    force: bool = False


class IndexJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"status": "idle", "processed": 0, "total": 0, "message": "尚未开始索引"}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def reserve(self):
        with self.lock:
            if self.state["status"] == "running":
                raise HTTPException(409, "已有索引任务正在运行")
            self.state = {"status": "running", "processed": 0, "total": 0, "message": "正在扫描照片"}

    def start(self, languages, force, library, model_dir):
        self.reserve()

        def worker():
            try:
                from app.indexer import build_index
                from app.clip_service import free_model
                library.sync()
                failures = 0
                for language in dict.fromkeys(languages):
                    self.update(processed=0, total=library.stats()["total"], message=f"正在更新{'中文' if language == 'zh' else '英文'}索引")
                    try:
                        build_index(lang=language, force=force, progress=lambda value: self.update(**value),
                                    image_dir=library.image_dir, index_dir=library.data_dir / "index", model_dir=model_dir)
                        failures += self.snapshot().get("failed", 0)
                    finally:
                        free_model(language, model_dir=model_dir)
                message = "索引更新完成" + (f"；{failures} 次图片处理失败，修复后可再次更新" if failures else "")
                self.update(status="completed", message=message)
            except Exception as exc:
                self.update(status="failed", message="索引未完成，请检查模型文件后重试", error=str(exc))

        threading.Thread(target=worker, name="album-index", daemon=True).start()
        return self.snapshot()


def create_app(*, data_dir: Path | None = None, model_dir: Path | None = None):
    library = Library(Path(data_dir or DATA_DIR))
    job = IndexJob()
    models = Path(model_dir or MODEL_DIR)
    model_paths = {"en": models / "clip-vit-base-patch32-local", "zh": models / "chinese-clip-vit-base-patch16-local"}

    @asynccontextmanager
    async def lifespan(application):
        await run_in_threadpool(library.sync)
        yield

    application = FastAPI(title="LUMA · 光屿智能相册", version="0.2.0", lifespan=lifespan)
    application.state.library = library
    application.state.model_paths = model_paths
    application.state.index_job = job

    @application.exception_handler(KeyError)
    async def not_found(request, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc.args[0])})

    @application.exception_handler(ValueError)
    async def invalid_input(request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    def model_status():
        return {language: {"available": (path / "config.json").is_file() and
                            (any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))),
                            "path": str(path)} for language, path in model_paths.items()}

    def index_status():
        result = {}
        for language, filename in (("en", "meta.json"), ("zh", "meta_zh.json")):
            path = library.data_dir / "index" / filename
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(meta, dict) or not isinstance(meta.get("paths"), list):
                    raise ValueError("无效的索引元数据")
                filename = meta.get("index_file") or ("images.index" if language == "en" else "images_zh.index")
                if Path(filename).name != filename or not (path.parent / filename).is_file():
                    raise ValueError("索引文件缺失")
                result[language] = {"count": len(meta["paths"]), "updated_at": meta.get("updated_at")}
            except (OSError, ValueError):
                result[language] = {"count": 0, "updated_at": None}
        return result

    def require_ai(language, *, index=False):
        name = "中文" if language == "zh" else "英文"
        if not model_status()[language]["available"]:
            raise HTTPException(503, f"尚未安装{name} CLIP 模型。可先使用关键词搜索，在模型就绪后更新索引。")
        missing = [package for package in ("torch", "transformers", "faiss") if importlib.util.find_spec(package) is None]
        if missing:
            raise HTTPException(503, "AI 依赖尚未安装，请先安装 requirements.txt 后重启服务。")
        if index and not index_status()[language]["count"]:
            raise HTTPException(503, f"尚未建立{name}索引，请在「智能引擎」中更新索引。")

    def translate_results(results, candidates):
        by_id = {photo["id"]: photo for photo in candidates}
        path_ids = library.by_path()
        return [{**by_id[path_ids[str(Path(result["path"]).resolve())]], "score": result["score"], "match_reason": "语义匹配"}
                for result in results if path_ids.get(str(Path(result["path"]).resolve())) in by_id]

    def execute_search(payload):
        started = time.perf_counter()
        photos = library.all(favorite=payload.favorite, album_id=payload.album_id, orientation=payload.orientation)
        keywords = keyword_rank(photos, payload.q)
        results, mode, warning = keywords, payload.mode, None
        if payload.mode != "keyword":
            language = "zh" if any("\u3400" <= char <= "\u9fff" for char in payload.q) else "en"
            try:
                require_ai(language, index=True)
                from app.retriever import search_by_text
                semantic = translate_results(search_by_text(payload.q, max(index_status()[language]["count"], 1),
                                                           index_dir=library.data_dir / "index", model_dir=models), photos)
                results = semantic if payload.mode == "semantic" else fuse_rankings(keywords, semantic)
            except (HTTPException, ImportError, OSError, RuntimeError, ValueError) as exc:
                detail = exc.detail if isinstance(exc, HTTPException) else "语义检索暂不可用，请检查模型和索引。"
                if payload.mode == "semantic":
                    raise HTTPException(503, detail) from exc
                mode, warning = "keyword", detail + " 本次已使用关键词搜索。"
        return {"items": results[:payload.k], "total": len(results), "mode": mode,
                "warning": warning, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}

    @application.get("/api/library")
    def list_photos(q: str = Query(default="", max_length=500), favorite: bool = False,
                    album_id: str | None = None, orientation: Orientation | None = None,
                    sort: Literal["newest", "oldest", "name"] = "newest",
                    page: int = Query(default=1, ge=1), page_size: int = Query(default=36, ge=1, le=100)):
        photos = library.all(favorite=favorite, album_id=album_id, orientation=orientation, sort=sort)
        if q.strip():
            photos = keyword_rank(photos, q)
        offset = (page - 1) * page_size
        return {"items": photos[offset:offset + page_size], "total": len(photos), "page": page, "page_size": page_size}

    @application.get("/api/stats")
    def stats():
        return library.stats()

    @application.post("/api/refresh")
    def refresh():
        return {"errors": library.sync(), **library.stats()}

    @application.get("/api/photos/{photo_id}")
    def photo(photo_id: str):
        return library.get(photo_id)

    @application.patch("/api/photos/{photo_id}")
    def update_photo(photo_id: str, payload: PhotoUpdate):
        return library.update(photo_id, payload.model_dump(exclude_unset=True, exclude_none=True))

    @application.get("/api/photos/{photo_id}/file")
    def original(photo_id: str):
        return FileResponse(library.path(photo_id))

    @application.get("/api/photos/{photo_id}/thumbnail")
    def thumbnail(photo_id: str):
        return FileResponse(library.thumbnail(photo_id), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    async def save_upload(file: UploadFile):
        temporary = tempfile.NamedTemporaryFile(dir=library.data_dir, prefix=".upload-", delete=False)
        path = Path(temporary.name)
        try:
            with temporary:
                size = 0
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        raise ValueError("每张图片最大 25 MB")
                    temporary.write(chunk)
            await run_in_threadpool(inspect_image, path)
            return path
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @application.post("/api/import")
    async def import_photos(files: Annotated[list[UploadFile], File()]):
        if len(files) > 50:
            for file in files:
                await file.close()
            raise HTTPException(400, "每次最多导入 50 张照片")
        result = {"imported": [], "duplicates": [], "errors": []}
        for file in files:
            path = None
            name = file.filename or "未命名照片"
            try:
                path = await save_upload(file)
                photo = await run_in_threadpool(library.import_file, path, name)
                result["imported" if photo else "duplicates"].append(photo if photo else name)
            except (ValueError, OSError) as exc:
                result["errors"].append({"name": name, "error": str(exc)})
            finally:
                if path:
                    path.unlink(missing_ok=True)
        return result

    @application.get("/api/albums")
    def albums():
        return {"items": library.albums()}

    @application.post("/api/albums", status_code=201)
    def create_album(payload: AlbumCreate):
        return library.create_album(payload.name, payload.description)

    @application.put("/api/albums/{album_id}/photos/{photo_id}")
    def add_to_album(album_id: str, photo_id: str):
        library.membership(album_id, photo_id)
        return {"ok": True}

    @application.delete("/api/albums/{album_id}/photos/{photo_id}")
    def remove_from_album(album_id: str, photo_id: str):
        library.membership(album_id, photo_id, add=False)
        return {"ok": True}

    @application.post("/api/search")
    def search(payload: SearchRequest):
        return execute_search(payload)

    @application.post("/api/photos/{photo_id}/similar")
    def similar(photo_id: str, payload: SimilarRequest):
        path = library.path(photo_id)
        require_ai("en", index=True)
        from app.retriever import search_by_image
        started = time.perf_counter()
        results = translate_results(call_ai(search_by_image, str(path), payload.k + 1,
                                            index_dir=library.data_dir / "index", model_dir=models), library.all())
        results = [photo for photo in results if photo["id"] != photo_id][:payload.k]
        return {"items": results, "total": len(results), "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}

    @application.post("/api/photos/{photo_id}/tags")
    def auto_tags(photo_id: str):
        path = library.path(photo_id)
        require_ai("en")
        from app.tagger import tag_image
        tags = call_ai(tag_image, str(path), 5, model_dir=models)
        existing = library.get(photo_id)["tags"]
        return library.update(photo_id, {"tags": list(dict.fromkeys(existing + [cn for _, cn, _ in tags]))[:30]})

    @application.get("/api/system")
    def system():
        return {"models": model_status(), "indexes": index_status(), "job": job.snapshot(), "device": "自动选择 · CUDA / MPS / CPU"}

    @application.post("/api/index", status_code=202)
    def update_indexes(payload: IndexRequest):
        for language in payload.languages:
            require_ai(language)
        return job.start(payload.languages, payload.force, library, models)

    @application.get("/health")
    def health():
        indexes = index_status()
        return {"status": "ok", "indexed_en": indexes["en"]["count"], "indexed_zh": indexes["zh"]["count"]}

    @application.post("/search")
    def legacy_search(q: str = Form(..., min_length=1, max_length=500), k: int = Form(default=12, ge=1, le=200)):
        result = execute_search(SearchRequest(q=q, k=k, mode="semantic"))
        return {"query": q, "results": [{"path": str(library.path(photo["id"])), "score": photo["score"]} for photo in result["items"]]}

    @application.post("/search_image")
    async def legacy_image_search(file: UploadFile = File(...), k: int = Form(default=12, ge=1, le=200)):
        require_ai("en", index=True)
        from app.retriever import search_by_image
        path = await save_upload(file)
        try:
            return {"results": await run_in_threadpool(call_ai, search_by_image, str(path), k,
                                                       index_dir=library.data_dir / "index", model_dir=models)}
        finally:
            path.unlink(missing_ok=True)

    @application.post("/tag")
    async def legacy_tag(file: UploadFile = File(...), topk: int = Form(default=3, ge=1, le=25)):
        require_ai("en")
        from app.tagger import tag_image
        path = await save_upload(file)
        try:
            tags = await run_in_threadpool(call_ai, tag_image, str(path), topk, model_dir=models)
            return {"tags": [{"en": en, "cn": cn, "score": score} for en, cn, score in tags]}
        finally:
            path.unlink(missing_ok=True)

    @application.post("/rebuild")
    def legacy_rebuild(force: bool = True):
        for language in ("en", "zh"):
            require_ai(language)
        job.reserve()
        try:
            from app.indexer import build_all_indexes
            library.sync()
            result = build_all_indexes(force=force, progress=lambda value: job.update(**value),
                                       image_dir=library.image_dir, index_dir=library.data_dir / "index", model_dir=models)
            job.update(status="completed", message="索引更新完成")
        except Exception as exc:
            job.update(status="failed", message="索引未完成", error=str(exc))
            raise HTTPException(503, "索引构建失败，请检查模型文件并重试") from exc
        return {"ok": True, **{f"indexed_{lang}": index.ntotal if index is not None else 0 for lang, index in result.items()}}

    @application.get("/image")
    def legacy_image(path: str):
        candidate = Path(path).resolve()
        if not candidate.is_relative_to(library.image_dir.resolve()):
            raise HTTPException(403, "只能访问本地相册中的照片")
        identifier = library.by_path().get(str(candidate))
        if not identifier:
            raise HTTPException(404, "照片不存在")
        return FileResponse(library.path(identifier))

    web_dir = BASE_DIR / "web"
    if web_dir.is_dir():
        application.mount("/assets", StaticFiles(directory=web_dir), name="assets")

    @application.get("/", include_in_schema=False)
    def home():
        return FileResponse(web_dir / "index.html")

    return application


app = create_app()
