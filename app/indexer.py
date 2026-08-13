# -*- coding: utf-8 -*-
"""建索引：把相册所有图片向量化，存入 FAISS。

中英双语各建一套索引（英文模型 / 中文模型生成的图片向量互不通用），
英文索引: images.index + meta.json，中文索引: images_zh.index + meta_zh.json
"""
import json
import time
import numpy as np
import faiss

from config import IMAGE_DIR, INDEX_DIR
from app.clip_service import embed_image_batch, free_model

INDEX_FILE = INDEX_DIR / "images.index"
META_FILE = INDEX_DIR / "meta.json"
INDEX_FILE_ZH = INDEX_DIR / "images_zh.index"
META_FILE_ZH = INDEX_DIR / "meta_zh.json"


def _index_files(lang="en"):
    if lang == "zh":
        return INDEX_FILE_ZH, META_FILE_ZH
    return INDEX_FILE, META_FILE


def _write_bytes_with_retry(path, data, tries=6):
    for i in range(tries):
        try:
            with open(path, "wb") as f:
                f.write(data)
            return True
        except PermissionError:
            time.sleep(1)
    return False


def _save_meta_with_retry(path, meta, tries=6):
    for i in range(tries):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
            return True
        except PermissionError:
            time.sleep(1)
    return False


def collect_images():
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted(
        p for p in IMAGE_DIR.rglob("*")
        if p.suffix.lower() in exts and not p.name.startswith(".")
    )
    return paths


def build_index(batch_size=16, force=False, lang="en"):
    """为 data/images 建一套指定语言的索引（en=英文CLIP, zh=中文CLIP）"""
    index_file, meta_file = _index_files(lang)
    if index_file.exists() and meta_file.exists() and not force:
        print(f"[indexer] 已有{lang}索引，跳过（加 --force 重建）")
        return load_index(lang)
    paths = collect_images()
    if not paths:
        print(f"[indexer] {IMAGE_DIR} 里没有图片，请先放图片。")
        return None
    name = "中文" if lang == "zh" else "英文"
    print(f"[indexer] 共 {len(paths)} 张图片，用{name}CLIP 向量化 ...")
    vectors, failed = embed_image_batch([str(p) for p in paths], batch_size, lang=lang)
    print(f"[indexer] 成功 {len(vectors)} 张，失败 {len(failed)} 张")
    if not vectors:
        return None
    vec = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(vec.shape[1])
    index.add(vec)
    if not _write_bytes_with_retry(index_file, faiss.serialize_index(index)):
        raise PermissionError(f"无法写入索引文件: {index_file}")
    ok_paths = [str(p) for p in paths]
    failed_set = {f[0] for f in failed}
    meta = {"paths": [p for p in ok_paths if p not in failed_set]}
    _save_meta_with_retry(meta_file, meta)
    print(f"[indexer] {name}索引已保存: {index_file} ({index.ntotal} 条)")
    return index


def build_all_indexes(batch_size=16, force=False):
    """同时构建中英两套索引（交替加载模型，省显存）"""
    result = {}
    result["en"] = build_index(batch_size=batch_size, force=force, lang="en")
    free_model("en")
    result["zh"] = build_index(batch_size=batch_size, force=force, lang="zh")
    return result


def load_index(lang="en"):
    index_file, meta_file = _index_files(lang)
    if not index_file.exists() or not meta_file.exists():
        return None
    with open(index_file, "rb") as f:
        data = f.read()
    index = faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["paths"]
