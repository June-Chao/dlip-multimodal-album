# -*- coding: utf-8 -*-
"""全局配置"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 数据目录
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"          # 相册图片放这里
INDEX_DIR = DATA_DIR / "index"           # 索引文件
MODEL_DIR = BASE_DIR / "models"          # 模型缓存（本地模型目录：clip-vit-base-patch32-local）

# CLIP 模型（本地路径，完全离线可用；英文效果好，检索关键词建议用英文或中英混合）
CLIP_MODEL_NAME = str(MODEL_DIR / "clip-vit-base-patch32-local")
# 中文 CLIP 模型（本地路径；中文查询自动切换到这个模型，中英文互不干扰）
CHINESE_CLIP_MODEL_NAME = str(MODEL_DIR / "chinese-clip-vit-base-patch16-local")

# Device detection stays lazy so the library browser can start without torch.
def get_device():
    """Honor FORCE_CPU/DEVICE, otherwise prefer CUDA, Apple MPS, then CPU."""
    if os.environ.get("FORCE_CPU", "").lower() in {"1", "true", "yes"}:
        return "cpu"
    requested = os.environ.get("DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested not in {"auto", "cuda", "mps"} and not (
        requested.startswith("cuda:") and requested[5:].isdigit()
    ):
        raise ValueError("DEVICE 必须是 auto、cpu、cuda、cuda:N 或 mps")
    import torch
    cuda_available = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    if requested.startswith("cuda"):
        if not cuda_available:
            raise RuntimeError("DEVICE 指定了 CUDA，但 CUDA 不可用；请使用 DEVICE=cpu 或 auto")
        if ":" in requested and int(requested.split(":", 1)[1]) >= torch.cuda.device_count():
            raise RuntimeError("DEVICE 指定的 CUDA 设备不存在")
        return requested
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("DEVICE 指定了 MPS，但当前设备不支持 MPS")
        return "mps"
    return "cuda" if cuda_available else ("mps" if mps_available else "cpu")


def __getattr__(name):
    # Preserve existing `from config import DEVICE` training/CLI callers.
    if name == "DEVICE":
        return get_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 检索默认返回数量
TOP_K = 12

# 相册零样本打标候选标签（英文给 CLIP 认，中文用于展示）
TAG_CANDIDATES = [
    ("person", "人像"), ("landscape", "风景"), ("food", "美食"),
    ("animal", "动物"), ("pet", "宠物"), ("building", "建筑"),
    ("city", "城市"), ("nature", "自然"), ("beach", "海滩"),
    ("mountain", "山"), ("forest", "森林"), ("flower", "花"),
    ("vehicle", "交通工具"), ("car", "汽车"), ("sport", "运动"),
    ("indoor", "室内"), ("outdoor", "户外"), ("night", "夜景"),
    ("sky", "天空"), ("sea", "大海"), ("baby", "宝宝"),
    ("wedding", "婚礼"), ("birthday", "生日"), ("travel", "旅行"),
    ("selfie", "自拍"),
]

for d in (IMAGE_DIR, INDEX_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)