# -*- coding: utf-8 -*-
"""Text and image retrieval in separate Chinese/English CLIP spaces."""
from pathlib import Path

import numpy as np

from config import TOP_K
from app.clip_service import embed_text, embed_image_file, detect_lang
from app.indexer import load_index


def _search(vector, k=TOP_K, lang="en", index_dir=None):
    if not isinstance(k, int) or k < 1:
        raise ValueError("返回数量必须是正整数")
    data = load_index(lang, index_dir=index_dir)
    if data is None:
        return []
    index, paths = data
    if index.ntotal == 0:
        return []
    query = np.ascontiguousarray(np.asarray(vector, dtype=np.float32).reshape(1, -1))
    if query.shape[1] != index.d or not np.isfinite(query).all():
        raise ValueError("查询向量与索引不兼容，请重新构建对应语言的索引")
    scores, ids = index.search(query, min(k, index.ntotal))
    return [{"path": paths[row], "score": round(float(score), 4)}
            for score, row in zip(scores[0], ids[0])
            if 0 <= row < len(paths) and Path(paths[row]).is_file()]


def search_by_text(text, k=TOP_K, *, index_dir=None, model_dir=None):
    """Automatically route a text query to its matching language model/index."""
    lang = detect_lang(text)
    model_options = {"model_dir": model_dir} if model_dir is not None else {}
    vector = embed_text(text, lang=lang, **model_options)
    return _search(vector, k, lang=lang, index_dir=index_dir)


def search_by_image(image_path, k=TOP_K, *, index_dir=None, model_dir=None):
    """Image similarity uses the English CLIP semantic space."""
    model_options = {"model_dir": model_dir} if model_dir is not None else {}
    vector = embed_image_file(image_path, lang="en", **model_options)
    return _search(vector, k, lang="en", index_dir=index_dir)
