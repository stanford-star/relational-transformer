from __future__ import annotations

import glob

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(
    "file_descriptor" not in torch.multiprocessing.get_all_sharing_strategies(),
    reason="file_system is the only sharing strategy on macOS and Windows",
)
def test__sharing_strategy__is_leak_proof() -> None:
    import rt.data.resolve  # noqa: F401  -- sets it at import

    assert torch.multiprocessing.get_sharing_strategy() == "file_descriptor", (
        "file_system leaks a named /dev/shm segment per shared tensor whenever "
        "a worker is killed rather than exited"
    )


def test__nofile_limit__is_raised_to_the_hard_limit() -> None:
    import resource

    import rt.data.resolve  # noqa: F401

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert soft == hard


@pytest.mark.slow
def test__dataloader_rebuilds__leave_no_named_segments() -> None:
    from torch.utils.data import DataLoader, Dataset

    import rt.data.resolve  # noqa: F401

    class Tiny(Dataset):
        def __len__(self) -> int:
            return 64

        def __getitem__(self, i: int) -> torch.Tensor:
            return torch.zeros(128, 128)

    before = set(glob.glob("/dev/shm/torch_*"))
    for _ in range(3):
        loader = DataLoader(Tiny(), batch_size=8, num_workers=2)
        assert sum(int(b.shape[0]) for b in loader) == 64
        del loader
    assert not set(glob.glob("/dev/shm/torch_*")) - before


def test__evaluator_entry__members_are_the_third_element() -> None:
    import inspect

    import rt.train._train as t

    src = inspect.getsource(t.main)
    assert "evaluators[0][2][0].mem_guard" in src, (
        "the guard must take a member (element 2), not a ctx size (element 1)"
    )
    unpack = [ln for ln in src.splitlines() if "in evaluators:" in ln and "for" in ln]
    assert any("tag, ctxs, members" in ln for ln in unpack), (
        "the entry shape changed; the guard's index has to change with it"
    )
