from dataclasses import dataclass
from pathlib import Path

CONFIGS_DIR = Path(__file__).parents[2] / "reproduce" / "tune"


@dataclass(frozen=True)
class Model:
    name: str
    ckpt_env: str
    grid_stem: str
    configs_path: str
    out_subdir: str
    zip_stem: str


MODELS = {
    model.name: model
    for model in (
        Model(
            name="rt-j",
            ckpt_env="RT_CKPT",
            grid_stem="tune--{db}--{table}",
            configs_path=str(CONFIGS_DIR / "tuned_configs_rt-j.json"),
            out_subdir="leaderboard",
            zip_stem="rt-j-icl",
        ),
        Model(
            name="rt-plurel",
            ckpt_env="RT_PLUREL_CKPT",
            grid_stem="tune-rt-plurel--{db}--{table}",
            configs_path=str(CONFIGS_DIR / "tuned_configs_rt-plurel.json"),
            out_subdir="leaderboard-rt-plurel",
            zip_stem="rt-plurel-icl",
        ),
    )
}


def model_from_argv(argv: list[str]) -> Model:
    assert len(argv) == 2 and argv[1] in MODELS, (
        f"usage: python -m examples.icl.<stage> <model>, where <model> is one of "
        f"{sorted(MODELS)}"
    )
    return MODELS[argv[1]]
