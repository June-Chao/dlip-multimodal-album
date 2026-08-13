# -*- coding: utf-8 -*-
"""生成微调前后效果对比图。"""
import sys, os
sys.path.insert(0, r"D:\项目\多模态相册")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for name in ["Microsoft YaHei", "SimHei"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

metrics = ["分类准确率", "宏平均准确率", "检索召回率@3"]
before = [86.96, 90.62, 79.17]
after = [91.30, 93.75, 90.62]

x = range(len(metrics)); w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar([i - w/2 for i in x], before, w, label="微调前（零样本 CLIP）", color="#8ecae6")
b2 = ax.bar([i + w/2 for i in x], after, w, label="微调后（CLIP + LoRA）", color="#219ebc")
for b in list(b1) + list(b2):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=10)
ax.set_xticks(list(x)); ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel("百分比 (%)", fontsize=12); ax.set_ylim(0, 110)
ax.set_title("CLIP LoRA 微调前后效果对比（8 类相册检索任务）", fontsize=13)
ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "效果对比.png"))
plt.tight_layout(); plt.savefig(out, dpi=150)
print("对比图已更新 ->", out)