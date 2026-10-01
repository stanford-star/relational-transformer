import hashlib
import json
import shutil
from pathlib import Path


def derange(indices: list[int], seed_material: str) -> dict[int, int]:
    import numpy as np

    assert len(indices) >= 2, f"need >= 2 columns to derange ({seed_material})"
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "little")
    )
    shuffled = list(indices)
    for i in range(len(shuffled) - 1, 0, -1):
        j = int(rng.integers(0, i))
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return dict(zip(indices, shuffled))


def relink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def main(*, pre_dir: str, out_dir: str, embedder: str, d_text: int, seed: int) -> None:
    import numpy as np

    pre = Path(pre_dir).expanduser()
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    relink(out / "db-task-lists", pre / "db-task-lists")

    pairs = json.loads((pre / "db-task-lists" / "forecast.json").read_text())
    for db in sorted({db for db, _ in pairs}):
        src = pre / db
        dst = out / db
        dst.mkdir(exist_ok=True)
        emb_name = f"text_emb_{embedder}.bin"
        for f in src.iterdir():
            if f.name == emb_name:
                continue
            relink(dst / f.name, f)

        col_index = json.loads((src / "column_index.json").read_text())
        perm = derange(sorted(col_index.values()), f"{db}:{seed}")

        shutil.copyfile(src / emb_name, dst / emb_name)
        src_mm = np.memmap(src / emb_name, dtype=np.uint16, mode="r").reshape(
            -1, d_text
        )
        dst_mm = np.memmap(dst / emb_name, dtype=np.uint16, mode="r+").reshape(
            -1, d_text
        )
        assert max(col_index.values()) < src_mm.shape[0]
        for orig, repl in perm.items():
            dst_mm[orig] = src_mm[repl]
        dst_mm.flush()
        print(f"{db}: deranged {len(perm)} column-name embeddings", flush=True)
