"""Download official model snapshots once; application inference stays offline."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="下载 CLIP 到项目 models 目录")
    parser.add_argument("--lang", choices=["en", "zh", "all"], default="all")
    args = parser.parse_args()
    from huggingface_hub import list_repo_files, snapshot_download

    root = Path(__file__).resolve().parents[1] / "models"
    models = {
        "en": ("openai/clip-vit-base-patch32", "clip-vit-base-patch32-local"),
        "zh": ("OFA-Sys/chinese-clip-vit-base-patch16", "chinese-clip-vit-base-patch16-local"),
    }
    for language in (["en", "zh"] if args.lang == "all" else [args.lang]):
        repository, folder = models[language]
        destination = root / folder
        files = list_repo_files(repository)
        weights = "*.safetensors" if any(name.endswith(".safetensors") for name in files) else "pytorch_model*.bin"
        snapshot_download(repo_id=repository, local_dir=destination,
                          allow_patterns=["*.json", "*.txt", weights])
        print(f"已下载 {repository} → {destination}")


if __name__ == "__main__":
    main()
