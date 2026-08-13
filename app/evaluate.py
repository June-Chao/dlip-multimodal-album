# -*- coding: utf-8 -*-
"""效果评估：分类准确率 + 检索 Recall@K。"""
import numpy as np
import torch
from PIL import Image

from app.dataset import category_prompts


@torch.no_grad()
def evaluate(model, processor, test_items, categories, k=3, device="cuda"):
    """用给定模型评估测试集。
    返回: accuracy, macro_accuracy, recall@k, 各类别明细
    """
    prompts = category_prompts(categories)

    # 1) 每个类别的文本向量（多模板取平均并归一化）
    text_vecs = {}
    for c in categories:
        inputs = processor(text=prompts[c], return_tensors="pt", padding=True).to(device)
        feats = model.get_text_features(**inputs).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        text_vecs[c] = feats.mean(dim=0)
    T = torch.stack([text_vecs[c] for c in categories])
    T = T / T.norm(dim=-1, keepdim=True)

    # 2) 测试图片向量
    img_vecs, labels = [], []
    for it in test_items:
        img = Image.open(it["path"]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        feat = model.get_image_features(**inputs).pooler_output
        feat = feat / feat.norm(dim=-1, keepdim=True)
        img_vecs.append(feat)
        labels.append(it["label"])
    I = torch.cat(img_vecs)

    # 3) 分类准确率
    sims = I @ T.T
    preds = [categories[i] for i in sims.argmax(dim=1).tolist()]
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)

    per_cat = {}
    for c in categories:
        idx = [i for i, l in enumerate(labels) if l == c]
        if idx:
            per_cat[c] = sum(preds[i] == c for i in idx) / len(idx)

    # 4) 检索 Recall@K（查询=每类文本，图库=测试集全部）
    recall = {}
    for c in categories:
        q = text_vecs[c] / text_vecs[c].norm()
        scores = I @ q
        topk = scores.topk(min(k, len(labels))).indices.tolist()
        relevant = [i for i, l in enumerate(labels) if l == c]
        recall[c] = sum(1 for i in topk if i in relevant) / max(1, len(relevant))
    recall_k = float(np.mean(list(recall.values())))

    macro_acc = float(np.mean(list(per_cat.values()))) if per_cat else float(acc)
    return {
        "accuracy": round(float(acc), 4),
        "macro_accuracy": round(macro_acc, 4),
        f"recall@{k}": round(recall_k, 4),
        "n_test": len(test_items),
        "per_category_acc": {c: round(v, 4) for c, v in per_cat.items()},
        "per_category_recall": {c: round(v, 4) for c, v in recall.items()},
    }