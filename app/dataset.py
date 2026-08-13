# -*- coding: utf-8 -*-
"""数据集：加载带标签照片，划分训练/测试集。"""
import json
import random
import time
from collections import Counter
from pathlib import Path

from config import IMAGE_DIR, DATA_DIR

SPLIT_FILE = DATA_DIR / "split.json"

# 参与微调的类别（带标签数量 >= 4 的才会用）
KEEP_LABELS = {"dog", "cat", "sport", "flower", "building", "car", "night", "beach", "mountain", "sunset", "city", "food"}

PROMPTS = [
    "a photo of a {c}",
    "a {c}",
    "a picture of a {c}",
    "photo of a {c}",
    "an image of a {c}",
]


def load_labeled(min_count=4):
    items = []
    for p in sorted(IMAGE_DIR.glob("*.jpg")):
        if p.name.startswith("_"):
            continue
        label = p.name.split("_")[0]
        if label in KEEP_LABELS:
            items.append({"path": str(p), "label": label})
    cnt = Counter(i["label"] for i in items)
    items = [i for i in items if cnt[i["label"]] >= min_count]
    return items


def split_train_test(items, test_ratio=0.3, seed=42):
    random.seed(seed)
    by_label = {}
    for it in items:
        by_label.setdefault(it["label"], []).append(it)
    train, test = [], []
    for label, lst in by_label.items():
        random.shuffle(lst)
        n_test = max(1, round(len(lst) * test_ratio))
        test += lst[:n_test]
        train += lst[n_test:]
    random.shuffle(train)
    random.shuffle(test)
    return train, test


def _write_json(path, data, tries=8):
    for i in range(tries):
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            return True
        except PermissionError:
            time.sleep(1)
    return False


def save_split(train, test):
    ok = _write_json(SPLIT_FILE, {"train": train, "test": test})
    if not ok:
        raise PermissionError(f"无法写入划分文件: {SPLIT_FILE}")
    print(f"[dataset] 划分已保存: 训练 {len(train)} / 测试 {len(test)} -> {SPLIT_FILE}")


def load_split():
    if not SPLIT_FILE.exists():
        return None
    data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    return data["train"], data["test"]


def get_categories(items):
    return sorted({i["label"] for i in items})


def category_prompts(categories):
    return {c: [t.format(c=c) for t in PROMPTS] for c in categories}