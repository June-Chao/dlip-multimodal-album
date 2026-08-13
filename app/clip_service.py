# -*- coding: utf-8 -*-
"""CLIP 模型封装：负责把图片和文字都变成"语义向量"。

支持中英双语：英文查询用 openai CLIP，中文查询自动切换 Chinese CLIP，
两个模型各自负责自己的语义空间，互不干扰。
"""
import torch
from PIL import Image
import numpy as np

from config import CLIP_MODEL_NAME, CHINESE_CLIP_MODEL_NAME, DEVICE, MODEL_DIR

_models = {}  # lang -> (model, processor)


def detect_lang(text: str) -> str:
    """检测查询语言：含 CJK 汉字即视为中文，否则按英文处理"""
    for c in text:
        if "一" <= c <= "鿿":
            return "zh"
    return "en"


def _load_model(lang: str):
    """按语言加载对应 CLIP 模型（首次加载后缓存）"""
    if lang == "zh":
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        path = CHINESE_CLIP_MODEL_NAME
        model = ChineseCLIPModel.from_pretrained(
            path, cache_dir=str(MODEL_DIR), use_safetensors=True).to(DEVICE)
        processor = ChineseCLIPProcessor.from_pretrained(path, cache_dir=str(MODEL_DIR))
    else:
        from transformers import CLIPModel, CLIPProcessor
        path = CLIP_MODEL_NAME
        model = CLIPModel.from_pretrained(
            path, cache_dir=str(MODEL_DIR), use_safetensors=True).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(path, cache_dir=str(MODEL_DIR))
    model.eval()
    print(f"[clip_service] 加载{'中文' if lang == 'zh' else '英文'}模型 {path} ... (device={DEVICE})")
    return model, processor


def get_model(lang: str = "en"):
    if lang not in _models:
        _models[lang] = _load_model(lang)
    return _models[lang]


def free_model(lang: str):
    """释放指定语言的模型，释放显存（在切换大批量任务时用）"""
    if lang in _models:
        del _models[lang]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[clip_service] 已释放{lang}模型显存")


def embed_image_file(image_path: str, lang: str = "en") -> np.ndarray:
    """单张图片 -> 归一化向量 (512维)"""
    model, processor = get_model(lang)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feat = model.get_image_features(**inputs).pooler_output
    return _normalize(feat)


def embed_image_batch(image_paths, batch_size=16, lang: str = "en"):
    """批量图片 -> (向量列表, 失败的路径列表)"""
    model, processor = get_model(lang)
    vectors, failed = [], []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        images = []
        for p in batch:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                failed.append((p, str(e)))
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(**inputs).pooler_output
        for j in range(len(images)):
            vectors.append(_normalize(feats[j:j + 1]))
    return vectors, failed


def embed_text(text: str, lang: str = None) -> np.ndarray:
    """文字 -> 归一化向量 (512维)；lang 为空时自动检测中英文"""
    if lang is None:
        lang = detect_lang(text)
    model, processor = get_model(lang)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        feat = model.get_text_features(**inputs).pooler_output
    return _normalize(feat)


def _normalize(feat: torch.Tensor) -> np.ndarray:
    v = feat.detach().cpu().numpy()
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return v.astype("float32")
