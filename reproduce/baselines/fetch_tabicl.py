import shutil
from pathlib import Path

from reproduce import config
from reproduce.baselines.rel2tab.tabicl_batched import (
    CLF_CHECKPOINT,
    HF_REPO,
    REG_CHECKPOINT,
)


def main() -> None:
    from huggingface_hub import hf_hub_download

    DEST = Path(config.share()) / "tabicl"
    DEST.mkdir(parents=True, exist_ok=True)
    for filename in (CLF_CHECKPOINT, REG_CHECKPOINT):
        out = DEST / filename
        if out.exists():
            print(f"{out} exists, skipping")
            continue
        path = hf_hub_download(repo_id=HF_REPO, filename=filename)
        shutil.copyfile(path, out)
        print(f"fetched {filename} -> {out}")


if __name__ == "__main__":
    main()
