import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    name: str
    target: str
    args: dict


def call(job: Job) -> None:
    module_name, func_name = job.target.split(":")
    func = getattr(importlib.import_module(module_name), func_name)
    func(**job.args)


def run_sequential(jobs: Sequence[Job]) -> None:
    for i, job in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {job.name}", flush=True)
        call(job)


def run(jobs: Sequence[Job], launcher: Callable[[Job], None]) -> None:
    for job in jobs:
        launcher(job)


def describe(jobs: Sequence[Job]) -> None:
    for job in jobs:
        print(f"{job.name}\t{job.target}")
    print(f"{len(jobs)} jobs", flush=True)
