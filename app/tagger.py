# -*- coding: utf-8 -*-
"""零样本打标：不用训练，用 CLIP 给图片自动贴标签。"""
import numpy as np
import torch
from PIL import Image

from config import TAG_CANDIDATES, DEVICE
from app.clip_service import get_model

EN2CN = {en: cn for en, cn in TAG_CANDIDATES}


def _tag_prompts():
    return [f"a photo of {en}" for en, _ in TAG_CANDIDATES]


def tag_image(image_path, topk=3):
    """返回 [(英文标签, 中文标签, 置信度), ...]"""
    model, processor = get_model()
    img = Image.open(image_path).convert("RGB")
    prompts = _tag_prompts()
    inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        sim = outputs.logits_per_image.detach().cpu().numpy()[0]  # 图片 vs 每个标签
    probs = np.exp(sim - sim.max()) / np.exp(sim - sim.max()).sum()
    idx = np.argsort(probs)[::-1][:topk]
    result = []
    for i in idx:
        en = TAG_CANDIDATES[i][0]
        result.append((en, EN2CN.get(en, en), round(float(probs[i]), 4)))
    return result
