# -*- coding: utf-8 -*-
"""CLIP LoRA 微调（图文对比学习）。"""
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, CLIPProcessor

from config import CLIP_MODEL_NAME, MODEL_DIR, DEVICE
from app.dataset import PROMPTS, category_prompts

ADAPTER_DIR = MODEL_DIR / "clip-lora-album"

# CLIP 使用的图像归一化参数
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


class AlbumDataset(Dataset):
    """每张训练图配多个文本模板，取图时随机抽一个模板，增强文本多样性。"""

    def __init__(self, items, categories, transform):
        self.items = items
        self.prompts = category_prompts(categories)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["path"]).convert("RGB")
        img = self.transform(img)
        text = random.choice(self.prompts[it["label"]])
        return img, text


def build_lora_model(r=16, alpha=32, dropout=0.1):
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
    config = LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=dropout,
    )
    model = get_peft_model(model, config)
    return model


def _clip_loss(model, img_feats, txt_feats):
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * (img_feats @ txt_feats.t())
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    return loss


def train(train_items, categories, epochs=25, lr=5e-5, batch_size=8, seed=0):
    random.seed(seed)
    torch.manual_seed(seed)

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = build_lora_model()
    model = model.to(DEVICE)
    model.print_trainable_parameters()

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    ds = AlbumDataset(train_items, categories, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                    drop_last=len(ds) >= batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dl) * epochs

    print(f"[finetune] 训练开始: {len(train_items)} 张图, {len(categories)} 类, "
          f"{epochs} epochs, batch={batch_size}, lr={lr}, 总步数≈{total_steps}", flush=True)
    model.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        for imgs, texts in dl:
            imgs = imgs.to(DEVICE)
            text_inputs = processor(text=list(texts), return_tensors="pt", padding=True).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                img_feats = model.get_image_features(pixel_values=imgs).pooler_output
                txt_feats = model.get_text_features(**text_inputs).pooler_output
                loss = _clip_loss(model, img_feats, txt_feats)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running / max(1, len(dl))
        print(f"  epoch {ep}/{epochs}  loss={avg:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    print(f"[finetune] LoRA 适配器已保存 -> {ADAPTER_DIR}")
    return model
