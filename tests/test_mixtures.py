import pytest

from rt.data import get_mixture, get_mixture_path, list_mixtures

COUNTS = {
    ("the-join", "all"): 13243,
    ("the-join", "rt-j"): 13243,
    ("the-join", "forecast"): 4098,
    ("the-join", "autocomplete"): 9145,
    ("relbench", "all"): 34,
    ("relbench", "forecast"): 21,
    ("relbench", "autocomplete"): 13,
    ("plurel", "all"): 116088,
    ("plurel", "autocomplete"): 116088,
    ("plurel", "forecast"): 0,
    ("plurel", "rt-plurel-train"): 86211,
}


def test_list_mixtures():
    assert sorted(COUNTS) == list_mixtures()


@pytest.mark.parametrize(("collection", "name"), sorted(COUNTS))
def test_mixture_counts(collection, name):
    pairs = get_mixture(collection, name)
    assert len(pairs) == COUNTS[(collection, name)]
    assert all(isinstance(db, str) and isinstance(task, str) for db, task in pairs)


def test_the_join_rt_j_databases():
    assert len({db for db, _ in get_mixture("the-join", "rt-j")}) == 523


def test_plurel_train_databases():
    assert len({db for db, _ in get_mixture("plurel", "rt-plurel-train")}) == 1900


def test_unknown_mixture_asserts():
    with pytest.raises(AssertionError):
        get_mixture_path("the-join", "no-such-mixture")
