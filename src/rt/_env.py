import os

import torch

_ALLOC_CONF = "expandable_segments:True"

_FR_BUFFER_SIZE = "20000"
_FR_DUMP_TEMP_FILE = (
    f"{os.environ.get('TMPDIR', '/tmp')}/nccl_trace_"
    f"job{os.environ.get('SLURM_JOB_ID', os.getpid())}_rank_"
)

_TORCH_LOGS = [
    "+torch._inductor.codecache",
    "+torch._functorch._aot_autograd.autograd_cache",
]

_DYNAMO_CACHE_SIZE_LIMIT = 16


def _omp_threads(num_workers: int) -> int:
    cpus = len(os.sched_getaffinity(0))
    return max(1, cpus // max(1, num_workers))


def _setup_env(num_workers: int = 0) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)

    os.environ.setdefault("TORCH_FR_BUFFER_SIZE", _FR_BUFFER_SIZE)
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    os.environ.setdefault("TORCH_FR_DUMP_TEMP_FILE", _FR_DUMP_TEMP_FILE)

    os.environ["TORCH_LOGS"] = ",".join(
        filter(None, [os.environ.get("TORCH_LOGS", ""), *_TORCH_LOGS])
    )

    torch._dynamo.config.cache_size_limit = _DYNAMO_CACHE_SIZE_LIMIT

    os.environ.setdefault("OMP_NUM_THREADS", str(_omp_threads(num_workers)))
    torch.set_num_threads(1)
