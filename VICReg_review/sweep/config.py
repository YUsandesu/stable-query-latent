"""Declarative sweep configuration: one sweep.yaml = the whole experiment.

Replaces the 40-flag command lines. A SweepConfig enumerates the combo grid and
produces a stable per-combo ``config_hash`` (identity of the trained artifact);
the ledger uses that hash to decide what 'done' actually covers on resume.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class GridConfig:
    output_dims: list
    latent_scales: list
    base_num_latents: int
    train_game_counts: list          # ints, or the string "all" for the full pool
    sample_fractions: list
    arms: list


@dataclass
class ModelConfig:
    latent_dim: int = 256
    reduce_hidden: list = field(default_factory=lambda: [128])
    expander_dim: int = 128
    expander_hidden: list = field(default_factory=lambda: [128])


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    seed: int = 42
    data_workers: int = 0            # parallel H5 read procs (0=auto=cores-1, cap 16; 1=serial)


@dataclass
class MemoryConfig:
    vram_safety: float = 0.85
    ram_safety: float = 0.8          # fraction of host RAM the full cache may use
    calib: str = "measure"           # measure | load | off


@dataclass
class ProbeConfig:
    every: int = 5
    start_epoch: int = 3


@dataclass
class DataSeedConfig:
    train_game_seed: int = 20260626
    anchors: list = field(default_factory=lambda: ["1091500", "1385380"])


@dataclass(frozen=True)
class Combo:
    output_dim: int
    latent_scale: float
    num_latents: int
    train_games: int                 # 0 == full pool
    view: float
    arm: str

    @property
    def games_label(self) -> str:
        return "all" if self.train_games <= 0 else f"{self.train_games:03d}"

    @property
    def latent_suffix(self) -> str:
        # Matches legacy run_data_view_sweep.latent_scale_label: empty for the
        # base 256 latents, else "_lat<NNN>x<scale>". Keeping combo dir names
        # identical to the legacy sweep means the existing eval / probe tooling
        # finds the new sweep's checkpoints unchanged.
        if int(self.num_latents) == 256 and math.isclose(float(self.latent_scale), 1.0, abs_tol=1e-9):
            return ""
        scale_text = f"{float(self.latent_scale):g}".replace(".", "p").replace("-", "m")
        return f"_lat{int(self.num_latents):03d}x{scale_text}"

    @property
    def combo_id(self) -> str:
        return (f"dim{self.output_dim:03d}_{self.arm}_n{self.games_label}"
                f"_view{round(self.view * 100):02d}{self.latent_suffix}")


def _as_int_or_zero(value) -> int:
    if isinstance(value, str) and value.strip().lower() == "all":
        return 0
    return int(value)


@dataclass
class SweepConfig:
    experiment: str
    h5: str
    out_dir: str
    grid: GridConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    data_seed: DataSeedConfig = field(default_factory=DataSeedConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "SweepConfig":
        return cls(
            experiment=d["experiment"],
            h5=d["h5"],
            out_dir=d["out_dir"],
            grid=GridConfig(**d["grid"]),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            memory=MemoryConfig(**d.get("memory", {})),
            probe=ProbeConfig(**d.get("probe", {})),
            data_seed=DataSeedConfig(**d.get("data_seed", {})),
        )

    @classmethod
    def load(cls, path) -> "SweepConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # lazy: only needed for YAML configs
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def num_latents_for(self, latent_scale: float) -> int:
        return max(1, int(round(self.grid.base_num_latents * float(latent_scale))))

    def iter_combos(self):
        """Yield every Combo. Order matches the legacy sweep so on-disk combo
        dirs / resume align: output_dim -> latent_scale -> train_games -> view -> arm."""
        for output_dim in self.grid.output_dims:
            for latent_scale in self.grid.latent_scales:
                num_latents = self.num_latents_for(latent_scale)
                for raw_games in self.grid.train_game_counts:
                    train_games = _as_int_or_zero(raw_games)
                    for view in self.grid.sample_fractions:
                        for arm in self.grid.arms:
                            yield Combo(
                                output_dim=int(output_dim),
                                latent_scale=float(latent_scale),
                                num_latents=int(num_latents),
                                train_games=int(train_games),
                                view=float(view),
                                arm=str(arm),
                            )

    def config_hash(self, combo: Combo) -> str:
        """Stable hash of everything that defines the trained artifact. If any of
        it changes, the ledger's 'done' for this combo no longer applies."""
        identity = {
            "combo": asdict(combo),
            "model": asdict(self.model),
            "epochs": self.train.epochs,
            "batch_size": self.train.batch_size,
            "seed": self.train.seed,
            "train_game_seed": self.data_seed.train_game_seed,
            "anchors": list(self.data_seed.anchors),
        }
        blob = json.dumps(identity, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def combo_count(self) -> int:
        return sum(1 for _ in self.iter_combos())


def example_config() -> dict:
    """The current cloud experiment, as a plain dict (dump to sweep.yaml)."""
    return {
        "experiment": "cloud_full_sweep_a100",
        "h5": "game_review_data/embedding_h5.h5",
        "out_dir": "VICReg_review/heads/cloud_full_sweep_a100",
        "grid": {
            "output_dims": [18, 36, 64, 72],
            "latent_scales": [1, 2, 4],
            "base_num_latents": 256,
            "train_game_counts": [50, 100, 200, 500, 1000, 1500, 2000, "all"],
            "sample_fractions": [0.8, 0.6, 0.4, 0.2],
            "arms": ["grl", "nogrl"],
        },
        "model": {"latent_dim": 256, "reduce_hidden": [128], "expander_dim": 128, "expander_hidden": [128]},
        "train": {"epochs": 30, "batch_size": 128, "seed": 42, "data_workers": 0},
        "memory": {"vram_safety": 0.85, "calib": "measure"},
        "probe": {"every": 5, "start_epoch": 3},
        "data_seed": {"train_game_seed": 20260626, "anchors": ["1091500", "1385380"]},
    }
