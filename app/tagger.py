# -*- coding: utf-8 -*-
"""Zero-shot image tags from the local English CLIP model."""
import numpy as np

from config import TAG_CANDIDATES, get_device
from app.clip_service import model_session, _open_rgb

EN2CN = {en: cn for en, cn in TAG_CANDIDATES}


def _tag_prompts():
    return [f"a photo of {en}" for en, _ in TAG_CANDIDATES]


def tag_image(image_path, topk=3, *, model_dir=None):
    """Return (English label, Chinese label, relative probability) tuples."""
    if topk < 1:
        return []
    with model_session("en", model_dir=model_dir) as (model, processor):
        import torch
        with _open_rgb(image_path) as image:
            inputs = processor(text=_tag_prompts(), images=image, return_tensors="pt",
                               padding=True, truncation=True).to(get_device())
            with torch.inference_mode():
                similarities = model(**inputs).logits_per_image.detach().float().cpu().numpy()[0]
    exponentials = np.exp(similarities - similarities.max())
    probabilities = exponentials / exponentials.sum()
    return [(TAG_CANDIDATES[row][0], TAG_CANDIDATES[row][1], round(float(probabilities[row]), 4))
            for row in np.argsort(probabilities)[::-1][:topk]]
