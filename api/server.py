# -*- coding: utf-8 -*-
"""FastAPI 服务：把检索能力封装成接口，方便部署和对接。"""
import tempfile, os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

from app.retriever import search_by_text, search_by_image
from app.tagger import tag_image
from app.indexer import build_all_indexes, load_index

app = FastAPI(title="多模态相册 API", version="0.1.0")


@app.get("/health")
def health():
    en = load_index("en")
    zh = load_index("zh")
    return {"status": "ok",
            "indexed_en": en[0].ntotal if en else 0,
            "indexed_zh": zh[0].ntotal if zh else 0}


@app.post("/search")
def search(q: str = Form(...), k: int = Form(12)):
    results = search_by_text(q, k)
    return {"query": q, "results": results}


@app.post("/search_image")
async def search_image(file: UploadFile = File(...), k: int = Form(12)):
    suffix = os.path.splitext(file.filename or "tmp.jpg")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await file.read())
        tmp = f.name
    try:
        results = search_by_image(tmp, k)
    finally:
        os.unlink(tmp)
    return {"results": results}


@app.post("/tag")
async def tag(file: UploadFile = File(...), topk: int = Form(3)):
    suffix = os.path.splitext(file.filename or "tmp.jpg")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await file.read())
        tmp = f.name
    try:
        tags = tag_image(tmp, topk)
    finally:
        os.unlink(tmp)
    return {"tags": [{"en": en, "cn": cn, "score": s} for en, cn, s in tags]}


@app.post("/rebuild")
def rebuild(force: bool = True):
    res = build_all_indexes(force=force)
    en = res.get("en")
    zh = res.get("zh")
    return {"ok": True,
            "indexed_en": en.ntotal if en else 0,
            "indexed_zh": zh.ntotal if zh else 0}


@app.get("/image")
def image(path: str):
    return FileResponse(path)
