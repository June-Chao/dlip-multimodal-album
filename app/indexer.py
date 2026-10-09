# -*- coding: utf-8 -*-
"""Language-specific, incremental FAISS indexes with atomic snapshot publication.

The metadata file is the commit point: it points to an immutable index generation.
Readers can keep using the previous generation while the next one is being built.
Legacy images.index + meta.json pairs remain readable.
"""
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from config import IMAGE_DIR, INDEX_DIR, CLIP_MODEL_NAME, CHINESE_CLIP_MODEL_NAME
from app.clip_service import embed_image_batch, free_model

INDEX_FILE = INDEX_DIR / "images.index"
META_FILE = INDEX_DIR / "meta.json"
INDEX_FILE_ZH = INDEX_DIR / "images_zh.index"
META_FILE_ZH = INDEX_DIR / "meta_zh.json"
_BUILD_LOCKS = {"en": threading.Lock(), "zh": threading.Lock()}


def _index_files(lang="en", index_dir=None):
    if lang not in _BUILD_LOCKS:
        raise ValueError("索引语言必须是 en 或 zh")
    if index_dir is not None:
        root = Path(index_dir)
        suffix = "_zh" if lang == "zh" else ""
        return root / f"images{suffix}.index", root / f"meta{suffix}.json"
    return (INDEX_FILE_ZH, META_FILE_ZH) if lang == "zh" else (INDEX_FILE, META_FILE)


def collect_images(image_dir=None):
    """Collect supported local photos, ignoring hidden files and external symlinks."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    root = Path(image_dir if image_dir is not None else IMAGE_DIR).resolve()
    return sorted({
        p.resolve() for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
        and p.resolve().is_relative_to(root)
    })


def _fingerprint(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _model_signature(lang, model_dir=None):
    """Invalidate reused vectors when the local model itself has been replaced."""
    root = Path(CHINESE_CLIP_MODEL_NAME if lang == "zh" else CLIP_MODEL_NAME)
    if model_dir is not None:
        root = Path(model_dir).resolve() / root.name
    files = {}
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"}:
                files[path.name] = _fingerprint(path)
    return {"path": str(root), "files": files}


def _load_snapshot(lang, index_dir=None):
    legacy_file, meta_file = _index_files(lang, index_dir)
    if not meta_file.exists():
        return None
    with meta_file.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("language", lang) != lang:
        raise ValueError("索引的语言与请求不匹配，请重新构建")
    filename = meta.get("index_file")
    if filename:
        # Snapshot pointers are local basenames, never arbitrary paths.
        if Path(filename).name != filename:
            raise ValueError("索引元数据包含无效的文件路径")
        index_file = meta_file.parent / filename
    else:
        index_file = legacy_file
    if not index_file.exists():
        if filename:
            raise ValueError("索引快照文件缺失，请重新构建索引")
        return None
    data = index_file.read_bytes()
    index = faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))
    paths = meta.get("paths")
    if not isinstance(paths, list) or index.ntotal != len(paths):
        raise ValueError("索引与图片目录不一致，请重新构建索引")
    return index, paths, meta


def _atomic_write(path, data):
    """Flush a complete file before exposing it through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publish(index, meta, lang, index_dir=None):
    legacy_file, meta_file = _index_files(lang, index_dir)
    generation = legacy_file.with_name(f"{legacy_file.stem}.{uuid.uuid4().hex}.index")
    _atomic_write(generation, faiss.serialize_index(index).tobytes())
    meta["index_file"] = generation.name
    try:
        _atomic_write(meta_file, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    except BaseException:
        generation.unlink(missing_ok=True)
        raise
    # Older immutable files are deliberately retained: an in-flight reader in
    # another process may still hold their metadata. They are never half-written.


def build_index(batch_size=16, force=False, lang="en", progress=None, *,
                image_dir=None, index_dir=None, model_dir=None):
    """Build or update one index; always return a FAISS index.

    Unchanged photos reuse their vectors, additions/changes are encoded, and
    deleted photos disappear from the published snapshot. ``force`` encodes all
    photos again. ``progress`` optionally receives dictionaries of job progress.
    """
    _index_files(lang, index_dir)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size 必须是正整数")
    with _BUILD_LOCKS[lang]:
        return _build_index(batch_size, force, lang, progress, image_dir, index_dir, model_dir)


def _build_index(batch_size, force, lang, progress, image_dir, index_dir, model_dir):
    try:
        previous = _load_snapshot(lang, index_dir)
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError):
        # A rebuild repairs corrupt/incomplete indexes without touching the
        # previous commit until a complete replacement is available.
        previous = None
    paths = [str(path) for path in collect_images(image_dir)]
    fingerprints, failed = {}, []
    for path in paths:
        try:
            fingerprints[path] = _fingerprint(path)
        except OSError as error:
            failed.append((path, str(error)))
    model_signature = _model_signature(lang, model_dir)
    old_index, old_paths, old_meta = previous or (None, [], {})
    old_rows = {path: i for i, path in enumerate(old_paths)}
    old_fingerprints = old_meta.get("fingerprints", {})
    can_reuse = not force and old_meta.get("model") == model_signature
    reused = {
        path for path, fingerprint in fingerprints.items()
        if can_reuse and path in old_rows and old_fingerprints.get(path) == fingerprint
    }
    pending = [path for path in paths if path in fingerprints and path not in reused]
    removed = len(set(old_paths) - set(paths))
    processed = len(reused) + len(failed)

    def report(message):
        if progress:
            progress({"processed": processed, "total": len(paths), "reused": len(reused),
                      "encoded": max(0, processed - len(reused) - len(failed)),
                      "removed": removed, "failed": len(failed), "message": message})

    report(f"复用 {len(reused)} 张图片，待处理 {len(pending)} 张")
    if (previous and not pending and not failed and paths == old_paths
            and len(reused) == len(paths) and not force):
        report("索引已是最新，无需重新计算")
        return old_index

    new_vectors = {}
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        model_options = {"model_dir": model_dir} if model_dir is not None else {}
        vectors, batch_failed = embed_image_batch(batch, batch_size=batch_size, lang=lang, **model_options)
        failed_paths = {str(item[0]) for item in batch_failed}
        successful = [path for path in batch if path not in failed_paths]
        if len(vectors) != len(successful):
            raise ValueError("图片向量数量与成功处理的图片数量不一致")
        failed.extend(batch_failed)
        for path, vector in zip(successful, vectors):
            try:
                if _fingerprint(path) != fingerprints[path]:
                    failed.append((path, "图片在索引过程中发生变化，请再次更新索引"))
                    continue
            except OSError as error:
                failed.append((path, str(error)))
                continue
            value = np.asarray(vector, dtype=np.float32).reshape(-1)
            if not np.isfinite(value).all() or value.size == 0:
                raise ValueError("模型生成了无效的图片向量")
            new_vectors[path] = value
        processed += len(batch)
        report(f"已处理 {processed}/{len(paths)} 张图片")

    valid_paths = [path for path in paths if path in reused or path in new_vectors]
    rows = [old_index.reconstruct(old_rows[path]) if path in reused else new_vectors[path]
            for path in valid_paths]
    dimension = len(rows[0]) if rows else (old_index.d if old_index is not None else 512)
    index = faiss.IndexFlatIP(dimension)
    if rows:
        index.add(np.ascontiguousarray(np.vstack(rows), dtype=np.float32))
    meta = {
        "version": 2, "language": lang, "paths": valid_paths,
        "fingerprints": {path: fingerprints[path] for path in valid_paths},
        "model": model_signature, "updated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(valid_paths), "failed": [{"path": str(path), "error": error} for path, error in failed],
    }
    _publish(index, meta, lang, index_dir)
    report(f"索引已更新：{index.ntotal} 张图片，失败 {len(failed)} 张")
    return index


def build_all_indexes(batch_size=16, force=False, progress=None, *,
                      image_dir=None, index_dir=None, model_dir=None):
    """Build both semantic spaces, releasing each model when its work is done."""
    result = {}
    for lang in ("en", "zh"):
        callback = (lambda event, language=lang: progress({**event, "lang": language})) if progress else None
        try:
            result[lang] = build_index(batch_size=batch_size, force=force, lang=lang, progress=callback,
                                       image_dir=image_dir, index_dir=index_dir, model_dir=model_dir)
        finally:
            free_model(lang, model_dir=model_dir)
    return result


def load_index(lang="en", index_dir=None):
    """Return ``(FAISS index, paths)`` from one committed snapshot, or None."""
    snapshot = _load_snapshot(lang, index_dir)
    return (snapshot[0], snapshot[1]) if snapshot else None
