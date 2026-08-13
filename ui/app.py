# -*- coding: utf-8 -*-
"""Gradio 界面：文字搜图 / 以图搜图 / 自动打标。"""
import os, tempfile
import gradio as gr

from app.retriever import search_by_text, search_by_image
from app.tagger import tag_image
from app.indexer import build_all_indexes, load_index, collect_images
from config import IMAGE_DIR, TOP_K


def _gallery(results):
    items = []
    for r in results:
        items.append((r["path"], f"{os.path.basename(r['path'])}  {r['score']}"))
    return items


def text_search(q, k):
    if not q:
        return [], "请输入搜索词（中英文都支持，如 a dog in the park / 沙滩上的一只狗）"
    results = search_by_text(q, int(k))
    return _gallery(results), f"搜到 {len(results)} 张"


def img_search(file, k):
    if file is None:
        return [], "请上传一张图片"
    results = search_by_image(file, int(k))
    return _gallery(results), f"找到 {len(results)} 张相似图片"


def auto_tag(file):
    if file is None:
        return "请上传一张图片"
    tags = tag_image(file, 3)
    lines = [f"{cn} ({en}) : {s*100:.1f}%" for en, cn, s in tags]
    return "\n".join(lines)


def status():
    n = len(collect_images())
    en = load_index("en")
    zh = load_index("zh")
    parts = []
    if en:
        parts.append(f"英文索引 {en[0].ntotal} 张")
    if zh:
        parts.append(f"中文索引 {zh[0].ntotal} 张")
    base = f"相册共 {n} 张"
    return base + (" / " + " / ".join(parts) if parts else "（尚未建索引）")


with gr.Blocks(title="多模态相册") as demo:
    gr.Markdown("# 📸 多模态相册\nCLIP 图文检索 + 自动打标（仅本地运行）")
    with gr.Row():
        st = gr.Textbox(value=status, label="索引状态", interactive=False)
        btn_rebuild = gr.Button("重建索引")
    with gr.Tab("文字搜图"):
        q = gr.Textbox(label="搜索词", placeholder="中英文都支持，例如: a dog in the park / 沙滩上的一只狗")
        k1 = gr.Slider(1, 30, value=TOP_K, step=1, label="返回数量")
        btn1 = gr.Button("搜索")
        out1 = gr.Gallery(label="结果")
        msg1 = gr.Textbox(label="信息")
        btn1.click(text_search, [q, k1], [out1, msg1])
    with gr.Tab("以图搜图"):
        up2 = gr.Image(type="filepath", label="上传图片")
        k2 = gr.Slider(1, 30, value=TOP_K, step=1, label="返回数量")
        btn2 = gr.Button("搜索相似图")
        out2 = gr.Gallery(label="结果")
        msg2 = gr.Textbox(label="信息")
        btn2.click(img_search, [up2, k2], [out2, msg2])
    with gr.Tab("自动打标"):
        up3 = gr.Image(type="filepath", label="上传图片")
        btn3 = gr.Button("识别标签")
        out3 = gr.Textbox(label="标签")
        btn3.click(auto_tag, [up3], [out3])
    btn_rebuild.click(lambda: ("重建完成" if build_all_indexes(force=True).get("en") else "失败"), None, st)

if __name__ == "__main__":
    demo.launch()
