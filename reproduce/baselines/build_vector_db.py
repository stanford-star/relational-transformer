import json
import time
from pathlib import Path


def build_all(
    *,
    db_task_list: str,
    pre_dir: str,
    features_root: str,
    features_subdir: str,
    vector_db_root: str,
    ivf_threshold: int,
) -> None:
    import faiss
    import numpy as np

    from reproduce.baselines.rel2tab.featurizer import table_offset_and_len
    from rt.data import resolve_db_task_list

    pairs = sorted(set(resolve_db_task_list(db_task_list)))
    for db, table in pairs:
        out_dir = Path(vector_db_root).expanduser() / db
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = out_dir / f"{table}.index"
        vectors_path = out_dir / f"{table}_vectors.bin"
        if index_path.exists() and vectors_path.exists():
            print(f"{db}/{table}: index exists, skipping", flush=True)
            continue

        feat_dir = Path(features_root).expanduser() / db / features_subdir
        with open(feat_dir / f"{table}_meta.json") as f:
            meta = json.load(f)
        min_offset, total_nodes = table_offset_and_len(pre_dir, db, table)
        assert (
            meta["min_offset"] == min_offset and meta["total_nodes"] == total_nodes
        ), (
            f"{db}/{table}: feature meta {meta} disagrees with table_info "
            f"(offset {min_offset}, nodes {total_nodes}); features are stale"
        )
        vectors = np.fromfile(
            feat_dir / f"{table}_vectors.bin", dtype=np.float32
        ).reshape(total_nodes, meta["n_features"])

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        vectors = vectors / norms

        num_nodes, dim = vectors.shape
        t0 = time.perf_counter()
        if num_nodes > ivf_threshold:
            nlist = min(max(int(4 * np.sqrt(num_nodes)), 16), 65536)
            nprobe = max(8, int(np.sqrt(nlist)))
            index = faiss.index_factory(
                dim, f"IVF{nlist},Flat", faiss.METRIC_INNER_PRODUCT
            )
            index.train(vectors[: min(nlist * 40, num_nodes)])
            index.add(vectors)
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
            index.own_invlists = False
            old_invlists = index.invlists
            new_invlists = faiss.OnDiskInvertedLists(
                index.nlist, index.code_size, str(out_dir / f"{table}.ivfdata")
            )
            new_invlists.merge_from(old_invlists, 0)
            index.replace_invlists(new_invlists, True)
            new_invlists.this.disown()
            del old_invlists
            kind = f"IVF{nlist},Flat (ondisk, nprobe={nprobe})"
        else:
            index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
            index.add(vectors)
            kind = "Flat"

        faiss.write_index(index, str(index_path))
        vectors.astype(np.float32).tofile(vectors_path)
        print(
            f"{db}/{table}: {num_nodes:,} x {dim} -> {kind} "
            f"in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
