# -*- coding: utf-8 -*-
"""检索：文字搜图 + 以图搜图。

文字搜图会自动检测查询语言：中文用中文 CLIP 检索，英文用英文 CLIP 检索，
各自命中对应的索引，互不干扰。以图搜图沿用英文索引。
"""
import numpy as np
from config import TOP_K
from app.clip_service import embed_text, embed_image_file, detect_lang
from app.indexer import load_index


def _search(vector, k=TOP_K, lang="en"):
    data = load_index(lang)
    if data is None:
        return []
    index, paths = data
    if index.ntotal == 0:
        return []
    k = min(k, index.ntotal)
    scores, ids = index.search(vector.reshape(1, -1), k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        results.append({"path": paths[idx], "score": round(float(score), 4)})
    return results


def search_by_text(text, k=TOP_K):
    """文字搜图：自动路由中英文模型和索引"""
    lang = detect_lang(text)
    vec = embed_text(text, lang=lang)
    return _search(vec, k, lang=lang)


def search_by_image(image_path, k=TOP_K):
    """以图搜图：沿用英文 CLIP 索引"""
    vec = embed_image_file(image_path, lang="en")
    return _search(vec, k, lang="en")
