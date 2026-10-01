import _thread
import contextlib
import os
import threading

import numpy as np
import orjson
import torch
from ml_dtypes import bfloat16
from sentence_transformers import SentenceTransformer

CHUNK = 1_000_000

WATCH_SECONDS = 10.0
WATCH_GRACE_SECONDS = 60.0


class WorkerDied(RuntimeError):
    pass


def _watch_workers(pool, stop: threading.Event) -> None:
    while not stop.wait(WATCH_SECONDS):
        dead = [p for p in pool["processes"] if p.exitcode is not None]
        if not dead:
            continue
        which = ", ".join(f"{p.name} (exit {p.exitcode})" for p in dead)
        print(
            f"!! {len(dead)} of {len(pool['processes'])} encoding workers died "
            f"[{which}] -- their chunks will never come back, so failing here "
            f"rather than waiting on them",
            flush=True,
        )
        _thread.interrupt_main()
        if not stop.wait(WATCH_GRACE_SECONDS):
            print("!! still alive after the interrupt; exiting hard", flush=True)
            os._exit(1)
        return


@contextlib.contextmanager
def _watched_pool(model: SentenceTransformer, devices: list[str]):
    pool = model.start_multi_process_pool(devices)
    stop = threading.Event()
    watchdog = threading.Thread(
        target=_watch_workers, args=(pool, stop), name="worker-watchdog", daemon=True
    )
    watchdog.start()
    try:
        yield pool
    except KeyboardInterrupt as e:
        if any(p.exitcode is not None for p in pool["processes"]):
            raise WorkerDied(
                "an encoding worker died mid-pool (see the traceback it printed "
                "above -- on this cluster it is usually an uncorrectable ECC "
                "error from a bad GPU)"
            ) from e
        raise
    finally:
        stop.set()
        for q in (pool["input"], pool["output"]):
            with contextlib.suppress(Exception):
                q.cancel_join_thread()
        with contextlib.suppress(Exception):
            model.stop_multi_process_pool(pool)


class TextEmbedder:
    def __init__(self, batch_size, embedder, device, chunk=CHUNK):
        device_type = torch.device(device).type
        self.model = SentenceTransformer(
            f"sentence-transformers/{embedder}",
            device=device,
            model_kwargs={
                "dtype": torch.bfloat16 if device_type == "cuda" else torch.float32,
            },
        )
        self.batch_size = batch_size
        self.chunk = chunk

    def __call__(self, text_list, device):
        if isinstance(device, list):
            with _watched_pool(self.model, device) as pool:
                out = [
                    self.model.encode_multi_process(
                        text_list[i : i + self.chunk],
                        pool,
                        batch_size=self.batch_size,
                        show_progress_bar=True,
                    ).astype(bfloat16)
                    for i in range(0, len(text_list), self.chunk)
                ]
            return np.concatenate(out) if out else np.empty((0, 0), dtype=bfloat16)

        out = []
        for i in range(0, len(text_list), self.chunk):
            piece = text_list[i : i + self.chunk]
            emb = self.model.encode(
                piece,
                batch_size=self.batch_size,
                convert_to_numpy=False,
                convert_to_tensor=True,
                show_progress_bar=True,
                device=device,
            )
            out.append(emb.to(torch.bfloat16).cpu().view(torch.int16).numpy())
            del emb
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        if not out:
            return np.empty((0, 0), dtype=bfloat16)
        stacked = np.concatenate(out)
        return stacked if stacked.dtype == bfloat16 else stacked.view(bfloat16)


def embed_texts(
    dataset_name,
    pre_dir: str,
    device,
    batch_size,
    embedder,
):
    if device is None:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            device = [f"cuda:{i}" for i in range(n)] if n > 1 else "cuda:0"
            print(f"Using device(s): {device}")
        else:
            device = "cpu"

    init_device = device[0] if isinstance(device, list) else device

    text_path = f"{pre_dir}/{dataset_name}/text.json"
    with open(text_path, "rb") as f:
        raw = f.read()
    text_list = orjson.loads(raw)
    del raw
    print(f"Loaded {len(text_list)} texts from {text_path}")

    text_embedder = TextEmbedder(batch_size, embedder, init_device)
    emb = text_embedder(text_list, device=device)

    emb_path = f"{pre_dir}/{dataset_name}/text_emb_{embedder}.bin"
    emb.tofile(emb_path)
    print(f"Wrote {emb.shape} {emb.dtype} to {emb_path}")
