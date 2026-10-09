# -*- coding: utf-8 -*-
"""命令行入口
用法:
  python main.py index          # 为 data/images 里的图片建索引
  python main.py search "..."   # 文字搜图
  python main.py tag <图片>     # 自动打标
  python main.py web            # 启动智能相册 (http://127.0.0.1:8000)
  python main.py api            # 同时提供网页与 FastAPI (/docs)
  python main.py ui             # 启动 Gradio 界面
  # ---- 第二阶段：CLIP 微调 ----
  python main.py split          # 划分训练/测试集
  python main.py baseline       # 微调前效果评估（零样本）
  python main.py finetune [N]   # LoRA 微调（N=epochs，默认15）
  python main.py eval_ft        # 微调后效果评估
  python main.py phase2 [N]     # 一键：划分+基线+微调+微调后评估
"""
import sys


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd = args[0]

    if cmd == "index":
        from app.indexer import build_all_indexes
        build_all_indexes(force="--force" in args)

    elif cmd == "search":
        from app.retriever import search_by_text
        q = args[1]
        for r in search_by_text(q):
            print(r["path"], r["score"])

    elif cmd == "tag":
        from app.tagger import tag_image
        for en, cn, s in tag_image(args[1]):
            print(f"{cn}({en}): {s*100:.1f}%")

    elif cmd in {"api", "web"}:
        import uvicorn
        uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=False)

    elif cmd == "ui":
        from ui.app import demo
        demo.launch()

    elif cmd == "split":
        from app.dataset import load_labeled, split_train_test, save_split
        items = load_labeled()
        print(f"[split] 带标签图片 {len(items)} 张")
        tr, te = split_train_test(items)
        save_split(tr, te)
        from collections import Counter
        print("训练集:", dict(Counter(i["label"] for i in tr)))
        print("测试集:", dict(Counter(i["label"] for i in te)))

    elif cmd == "baseline":
        from app.clip_service import get_model
        from app.evaluate import evaluate
        from app.dataset import load_split, get_categories
        model, processor = get_model()
        split = load_split()
        if not split:
            print("请先运行 python main.py split")
            return
        tr, te = split
        cats = get_categories(tr + te)
        res = evaluate(model, processor, te, cats, k=3)
        print("\n===== 微调前（零样本 CLIP）=====")
        print(res)

    elif cmd == "finetune":
        from app.dataset import load_split, get_categories
        from app.finetune import train as train_clip
        split = load_split()
        if not split:
            print("请先运行 python main.py split")
            return
        tr, te = split
        cats = get_categories(tr + te)
        epochs = int(args[1]) if len(args) > 1 and args[1].isdigit() else 25
        train_clip(tr, cats, epochs=epochs)

    elif cmd == "eval_ft":
        from app.evaluate import evaluate
        from app.dataset import load_split, get_categories
        from app.finetune import ADAPTER_DIR
        from peft import PeftModel
        from transformers import CLIPModel, CLIPProcessor
        from config import CLIP_MODEL_NAME, DEVICE
        split = load_split()
        if not split:
            print("请先运行 python main.py split")
            return
        tr, te = split
        cats = get_categories(tr + te)
        base = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        model = model.to(DEVICE)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        res = evaluate(model, processor, te, cats, k=3)
        print("\n===== 微调后（CLIP + LoRA）=====")
        print(res)

    elif cmd == "phase2":
        from app.dataset import load_labeled, split_train_test, save_split, get_categories
        from app.clip_service import get_model
        from app.evaluate import evaluate
        from app.finetune import train as train_clip, ADAPTER_DIR
        from peft import PeftModel
        from transformers import CLIPModel, CLIPProcessor
        from config import CLIP_MODEL_NAME, DEVICE
        from collections import Counter

        items = load_labeled()
        print(f"[phase2] 带标签图片 {len(items)} 张")
        tr, te = split_train_test(items)
        save_split(tr, te)
        print("训练集:", dict(Counter(i["label"] for i in tr)))
        print("测试集:", dict(Counter(i["label"] for i in te)))
        cats = get_categories(tr + te)

        model, processor = get_model()
        before = evaluate(model, processor, te, cats, k=3)
        print("\n===== 微调前（零样本 CLIP）=====")
        print(before)

        epochs = int(args[1]) if len(args) > 1 and args[1].isdigit() else 25
        train_clip(tr, cats, epochs=epochs)

        base = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
        model_ft = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).to(DEVICE)
        after = evaluate(model_ft, processor, te, cats, k=3)
        print("\n===== 微调后（CLIP + LoRA）=====")
        print(after)

        print("\n===== 对比汇总 =====")
        for k in ["accuracy", "macro_accuracy", "recall@3"]:
            d = round((after[k] - before[k]) * 100, 2)
            print(f"{k}: {before[k]*100:.1f}% -> {after[k]*100:.1f}%  (变化 {d:+.2f} 个百分点)")

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
