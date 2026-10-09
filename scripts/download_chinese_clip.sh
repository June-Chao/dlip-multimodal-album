#!/usr/bin/env bash
# 下载中文 CLIP 模型（OFA-Sys/chinese-clip-vit-base-patch16）到本地，供离线使用
# 用 curl 直连 hf-mirror（huggingface_hub 在部分网络环境会卡住），支持断点续传
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$PROJECT_DIR/models/chinese-clip-vit-base-patch16-local"
mkdir -p "$DEST"

BASE="https://hf-mirror.com/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main"
TBASE="https://hf-mirror.com/bert-base-chinese/resolve/main"

dl() { # $1=url  $2=目标文件
  if [ -s "$2" ]; then
    echo "[skip] 已存在: $2"
    return
  fi
  echo "[下载] $1"
  curl -fSL -C - --retry 5 --retry-delay 3 -o "$2.part" "$1"
  mv "$2.part" "$2"
  echo "[完成] $2 ($(du -h "$2" | cut -f1))"
}

echo "===== 下载中文 CLIP 模型到 $DEST ====="
dl "$BASE/config.json"              "$DEST/config.json"
dl "$BASE/preprocessor_config.json" "$DEST/preprocessor_config.json"
dl "$BASE/vocab.txt"                "$DEST/vocab.txt"
dl "$BASE/pytorch_model.bin"        "$DEST/pytorch_model.bin"
# 补充 BERT 分词器配置（官方 chinese-clip 仓库未自带，用 bert-base-chinese 的）
dl "$TBASE/tokenizer_config.json"   "$DEST/tokenizer_config.json"
dl "$TBASE/special_tokens_map.json" "$DEST/special_tokens_map.json"

echo "===== 全部下载完成 ====="
ls -la "$DEST"
