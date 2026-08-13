# -*- coding: utf-8 -*-
"""下载演示相册数据：用系统 curl 从维基共享资源按类别下载通用照片。
（说明：本机 Python 直连维基会被网络重置，curl 正常，故用 curl 下载）
v3：小批量 + 翻页（gcmcontinue），解决大类别超时问题。

用法:
  python scripts/download_demo_data.py
  NUM_PER=15 TOTAL=350 python scripts/download_demo_data.py
"""
import json
import os
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "data" / "images"
MANIFEST_FILE = BASE_DIR / "data" / "manifest.json"

API = "https://commons.wikimedia.org/w/api.php"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

CATEGORIES = [
    ("Category:Dogs", "dog"),
    ("Category:Cats", "cat"),
    ("Category:Food", "food"),
    ("Category:Landscape photography", "landscape"),
    ("Category:Buildings", "building"),
    ("Category:Beaches", "beach"),
    ("Category:Mountains", "mountain"),
    ("Category:Flowers", "flower"),
    ("Category:Cars", "car"),
    ("Category:Sports", "sport"),
    ("Category:City streets", "city"),
    ("Category:Sunset", "sunset"),
    ("Category:Wedding", "wedding"),
    ("Category:Baby", "baby"),
    ("Category:Travel photography", "travel"),
    ("Category:Night", "night"),
    ("Category:Food photography", "food"),
    ("Category:Sunsets", "sunset"),
    ("Category:Cityscapes", "city"),
    ("Category:Forests", "forest"),
]


def curl_json(url, timeout=90):
    r = subprocess.run(
        ["curl.exe", "-s", "--max-time", str(timeout), "-A", UA, url],
        capture_output=True,
    )
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"curl 失败(code={r.returncode})")
    return json.loads(r.stdout.decode("utf-8", errors="ignore"))


def is_valid_jpeg(path):
    try:
        with open(path, "rb") as f:
            magic = f.read(3)
        return magic == b"\xff\xd8\xff" and path.stat().st_size > 10000
    except Exception:
        return False


def curl_download(url, path, timeout=90):
    clean_url = url.split("?")[0]
    r = subprocess.run(
        ["curl.exe", "-sL", "--max-time", str(timeout), "-A", UA, "-o", str(path), clean_url],
        capture_output=True,
    )
    if r.returncode != 0 or not path.exists() or not is_valid_jpeg(path):
        path.unlink(missing_ok=True)
        return False
    return True


def fetch_category_files(category, limit):
    """小批量(10) + 翻页，直到凑够 limit 张有效 jpeg 或取完。"""
    import urllib.parse
    files, cont = [], {}
    while len(files) < limit:
        params = {
            "action": "query", "generator": "categorymembers",
            "gcmtitle": category, "gcmtype": "file", "gcmlimit": 10,
            "prop": "imageinfo", "iiprop": "url|size|mime", "iiurlwidth": 600,
            "format": "json",
        }
        params.update(cont)
        url = API + "?" + urllib.parse.urlencode(params)
        data = curl_json(url)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            w, h = info.get("width", 0), info.get("height", 0)
            u = info.get("thumburl") or info.get("url")
            if mime == "image/jpeg" and w >= 400 and h >= 400:
                files.append(u)
                if len(files) >= limit:
                    break
        cont = data.get("continue", {})
        if not cont:
            break
        time.sleep(0.3)
    return files


def main(num_per=15, total_cap=350):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    total = 0
    for category, tag in CATEGORIES:
        if total >= total_cap:
            break
        urls = None
        for attempt in range(5):
            try:
                urls = fetch_category_files(category, num_per)
                break
            except Exception as e:
                print(f"  [{category}] 第{attempt+1}次失败: {str(e)[:100]}", flush=True)
                time.sleep(3)
        if not urls:
            print(f"[跳过] {category}", flush=True)
            continue
        got = 0
        for url in urls:
            if total >= total_cap or got >= num_per:
                break
            fname = f"{tag}_{category.split(':')[1].replace(' ', '_')}_{got+1:02d}.jpg"
            path = IMAGE_DIR / fname
            if path.exists():
                if is_valid_jpeg(path):
                    got += 1; total += 1
                    manifest.append({"image": fname, "category_en": tag, "category_cn": tag, "url": url})
                    continue
                else:
                    path.unlink(missing_ok=True)
            if curl_download(url, path):
                manifest.append({"image": fname, "category_en": tag, "category_cn": tag, "url": url})
                got += 1; total += 1
            time.sleep(0.2)
        print(f"[{category}] 下载 {got} 张（累计 {total}）", flush=True)
        time.sleep(1)

    old = []
    if MANIFEST_FILE.exists():
        try:
            old = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        except Exception:
            old = []
    old_picsum = [p for p in old if p["image"].startswith("picsum_")]
    manifest += old_picsum
    seen, uniq = set(), []
    for p in manifest:
        if p["image"] not in seen:
            seen.add(p["image"])
            uniq.append(p)
    MANIFEST_FILE.write_text(json.dumps(uniq, ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(list(IMAGE_DIR.glob("*.jpg")))
    print(f"\n完成：共 {total} 张带标签照片，相册现有 {n} 张", flush=True)


if __name__ == "__main__":
    main(
        num_per=int(os.environ.get("NUM_PER", 15)),
        total_cap=int(os.environ.get("TOTAL", 350)),
    )
