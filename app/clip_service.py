# -*- coding: utf-8 -*-
"""Offline CLIP inference with separate Chinese and English semantic spaces."""
from contextlib import contextmanager
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageOps

from config import CLIP_MODEL_NAME, CHINESE_CLIP_MODEL_NAME, MODEL_DIR, get_device

_models = {}
_MODEL_LOCK = threading.RLock()


def detect_lang(text: str) -> str:
    """Queries containing CJK ideographs use Chinese CLIP."""
    return "zh" if any("\u3400" <= char <= "\u9fff" or "\U00020000" <= char <= "\U0002fa1f" for char in text) else "en"


def _model_path(lang, model_dir=None):
    if lang not in {"en", "zh"}:
        raise ValueError("模型语言必须是 en 或 zh")
    default = Path(CHINESE_CLIP_MODEL_NAME if lang == "zh" else CLIP_MODEL_NAME)
    return (Path(model_dir) / default.name if model_dir is not None else default).resolve()


def _load_model(lang: str, model_dir=None):
    root = _model_path(lang, model_dir)
    path = str(root)
    cache_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    safetensors = any(file.is_file() for file in root.glob("*.safetensors"))
    legacy_weights = any(file.is_file() for file in root.glob("pytorch_model*.bin"))
    if not root.is_dir() or not (root / "config.json").is_file() or not (safetensors or legacy_weights):
        raise FileNotFoundError(f"未找到{'中文' if lang == 'zh' else '英文'} CLIP 本地模型：{path}。请先按 README 下载模型。")
    if lang == "zh":
        from transformers import ChineseCLIPModel as Model, ChineseCLIPProcessor as Processor
    else:
        from transformers import CLIPModel as Model, CLIPProcessor as Processor
    device = get_device()
    try:
        model = Model.from_pretrained(path, cache_dir=str(cache_dir),
                                      use_safetensors=safetensors, local_files_only=True).to(device)
        processor = Processor.from_pretrained(path, cache_dir=str(cache_dir), local_files_only=True)
    except OSError as error:
        raise FileNotFoundError(f"本地 CLIP 模型文件不完整：{path}。请重新下载模型。") from error
    model.eval()
    print(f"[clip_service] 已加载 {lang} 模型 (device={device})")
    return model, processor


def get_model(lang: str = "en", model_dir=None):
    key = (lang, str(_model_path(lang, model_dir)))
    with _MODEL_LOCK:
        if key not in _models:
            _models[key] = _load_model(lang, model_dir=model_dir)
        return _models[key]


@contextmanager
def model_session(lang="en", model_dir=None):
    """Keep loading, inference and freeing mutually exclusive across web jobs."""
    with _MODEL_LOCK:
        yield get_model(lang, model_dir=model_dir)


def free_model(lang: str, model_dir=None):
    key = (lang, str(_model_path(lang, model_dir)))
    with _MODEL_LOCK:
        if key not in _models:
            return
        del _models[key]
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def _open_rgb(image_path):
    """Detach RGB pixels from the file and honor phone-camera EXIF rotation."""
    with Image.open(image_path) as source:
        with ImageOps.exif_transpose(source) as oriented:
            return oriented.convert("RGB")


def _feature_tensor(output):
    """Transformers versions return either tensors or structured features."""
    if hasattr(output, "detach"):
        return output
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    raise TypeError("CLIP 特征输出格式不受支持，请检查 transformers 版本")


def embed_image_file(image_path: str, lang: str = "en", model_dir=None) -> np.ndarray:
    with model_session(lang, model_dir=model_dir) as (model, processor):
        import torch
        with _open_rgb(image_path) as image:
            inputs = processor(images=image, return_tensors="pt").to(get_device())
            with torch.inference_mode():
                features = _feature_tensor(model.get_image_features(**inputs))
            return _normalize(features)


def embed_image_batch(image_paths, batch_size=16, lang: str = "en", model_dir=None):
    """Return successful vectors in input order and ``(path, error)`` failures."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size 必须是正整数")
    if not image_paths:
        return [], []
    # Check local model files before importing the optional inference runtime.
    with model_session(lang, model_dir=model_dir) as (model, processor):
        import torch
        vectors, failed = [], []
        for offset in range(0, len(image_paths), batch_size):
            images = []
            try:
                for path in image_paths[offset:offset + batch_size]:
                    try:
                        images.append(_open_rgb(path))
                    except (OSError, ValueError, Image.DecompressionBombError) as error:
                        failed.append((path, str(error)))
                if not images:
                    continue
                inputs = processor(images=images, return_tensors="pt").to(get_device())
                with torch.inference_mode():
                    features = _feature_tensor(model.get_image_features(**inputs))
                vectors.extend(_normalize(features[row:row + 1]) for row in range(len(images)))
            finally:
                for image in images:
                    image.close()
        return vectors, failed


def embed_text(text: str, lang: str = None, model_dir=None) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("搜索内容不能为空")
    lang = lang or detect_lang(text)
    with model_session(lang, model_dir=model_dir) as (model, processor):
        import torch
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(get_device())
        with torch.inference_mode():
            features = _feature_tensor(model.get_text_features(**inputs))
        return _normalize(features)


def _normalize(features) -> np.ndarray:
    value = features.detach().float().cpu().numpy()
    value = value / (np.linalg.norm(value, axis=-1, keepdims=True) + 1e-9)
    return value.astype("float32")
