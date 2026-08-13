# -*- coding: utf-8 -*-
"""补充下载：从 picsum.photos 批量下载通用照片（稳定源）。

用法:
  python scripts/download_picsum.py            # 默认 120 张
  TOTAL=200 python scripts/download_picsum.py
"""
import json
import os
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "data" / "images"
MANIFEST_FILE = BASE_DIR / "data" / "manifest.json"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def curl_get(url, timeout=30):
    r = subprocess.run(
        ["curl.exe", "-s", "--max-time", str(timeout), "-A", UA, url],
        capture_output=True,
    )
    return r.stdout


def is_valid_jpeg(path):
    try:
        with open(path, "rb") as f:
            return f.read(3) == b"\xff\xd8\xff" and path.stat().st_size > 10000
    except Exception:
        return False


def main(total=120):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    have = {p["image"] for p in manifest}

    ids = []
    for page in range(1, 12):  # 前 11 页，约 330 个候选
        try:
            raw = curl_get(f"https://picsum.photos/v2/list?page={page}&limit=30")
            data = json.loads(raw.decode("utf-8", errors="ignore"))
            ids += [str(d["id"]) for d in data]
        except Exception:
            pass
        time.sleep(0.3)
    print(f"候选 id 数: {len(ids)}", flush=True)

    added = 0
    for pid in ids:
        if added >= total:
            break
        fname = f"picsum_{pid}.jpg"
        path = IMAGE_DIR / fname
        if fname in have and path.exists() and is_valid_jpeg(path):
            continue
        r = subprocess.run(
            ["curl.exe", "-sL", "--max-time", "40", "-A", UA,
             "-o", str(path), f"https://picsum.photos/id/{pid}/500/500"],
            capture_output=True,
        )
        if r.returncode == 0 and is_valid_jpeg(path):
            manifest.append({"image": fname, "category_en": "picsum", "category_cn": "picsum", "url": f"https://picsum.photos/id/{pid}"})
            added += 1
            if added % 20 == 0:
                print(f"已补充 {added}/{total}", flush=True)
        else:
            path.unlink(missing_ok=True)
        time.sleep(0.1)

    # 去重并写回清单
    seen = set()
    uniq = []
    for p in manifest:
        if p["image"] not in seen:
            seen.add(p["image"])
            uniq.append(p)
    MANIFEST_FILE.write_text(json.dumps(uniq, ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(list(IMAGE_DIR.glob("*.jpg")))
    print(f"完成：本次新增 {added} 张，相册现有 {n} 张", flush=True)


if __name__ == "__main__":
    main(total=int(os.environ.get("TOTAL", 120)))