from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .registry import get_model_spec
from .util import deep_merge, expand_path, safe_slug, set_dotted, stable_hash


DEFAULT_TRAINING: dict[str, Any] = {
    "epochs": 30,
    "batch_size": 2,
    "learning_rate": 5.0e-5,
    "weight_decay": 1.0e-5,
    "segment_seconds": 4.0,
    "validation_segment_seconds": 4.0,
    "num_workers": 4,
    "gradient_accumulation": 1,
    "gradient_clip": 5.0,
    "precision": "fp16",
    "patience": 8,
    "lr_patience": 2,
    "lr_factor": 0.5,
    "min_learning_rate": 1.0e-8,
    "use_speed_perturb": False,
    "use_wavedrop": False,
}


@dataclass(frozen=True)
class ResolvedJob:
    model: str
    speakers: int
    repeat: int
    seed: int
    backend: str
    sample_rate: int
    training: dict[str, Any]
    initial_checkpoint_value: str | None
    sweep_values: dict[str, Any]
    run_hash: str
    run_label: str

    def identity(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "speakers": self.speakers,
            "repeat": self.repeat,
            "seed": self.seed,
            "backend": self.backend,
            "sample_rate": self.sample_rate,
            "training": self.training,
            "initial_checkpoint": self.initial_checkpoint_value,
            "sweep": self.sweep_values,
        }


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    validate_config(raw)
    return raw, config_path


def validate_config(config: dict[str, Any]) -> None:
    version = config.get("version")
    if version != 1:
        raise ValueError(f"Unsupported config version {version!r}; expected 1")
    experiment = config.get("experiment")
    if not isinstance(experiment, dict) or not experiment.get("name"):
        raise ValueError("experiment.name is required")
    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("root"):
        raise ValueError("dataset.root is required")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    for entry in models:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError("Every models entry requires name")
        counts = entry.get("speakers")
        if isinstance(counts, int):
            counts = [counts]
        if not isinstance(counts, list) or not counts:
            raise ValueError(f"models.{entry.get('name')}.speakers must be an integer or list")
        for count in counts:
            get_model_spec(str(entry["name"]), int(count))
        _validate_sweep(entry.get("sweep", {}), prefix=f"models.{entry['name']}.sweep")
    repeats = int(experiment.get("repeats", 1))
    if repeats <= 0:
        raise ValueError("experiment.repeats must be positive")
    _validate_training(deep_merge(DEFAULT_TRAINING, config.get("training", {})))
    _validate_sweep(config.get("sweep", {}), prefix="sweep")


def _validate_sweep(sweep: Any, prefix: str) -> None:
    if not sweep:
        return
    if not isinstance(sweep, dict) or not isinstance(sweep.get("grid", {}), dict):
        raise ValueError(f"{prefix}.grid must be a mapping of dotted keys to lists")
    for key, values in sweep.get("grid", {}).items():
        if not str(key).startswith("training."):
            raise ValueError(f"{prefix} key must begin with 'training.': {key}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{prefix} values must be a non-empty list: {key}")


def _validate_training(training: dict[str, Any]) -> None:
    positive = (
        "epochs", "batch_size", "learning_rate", "segment_seconds", "num_workers",
        "gradient_accumulation",
    )
    for key in positive:
        value = training.get(key)
        if value is None or float(value) <= 0 and key != "num_workers":
            raise ValueError(f"training.{key} must be positive")
        if key == "num_workers" and int(value) < 0:
            raise ValueError("training.num_workers cannot be negative")
    if training.get("precision") not in {"fp32", "fp16", "bf16"}:
        raise ValueError("training.precision must be fp32, fp16, or bf16")
    if float(training.get("weight_decay", 0)) < 0:
        raise ValueError("training.weight_decay cannot be negative")
    if float(training.get("validation_segment_seconds", training["segment_seconds"])) <= 0:
        raise ValueError("training.validation_segment_seconds must be positive")
    if int(training.get("patience", 1)) <= 0:
        raise ValueError("training.patience must be positive")
    if int(training.get("lr_patience", 0)) < 0:
        raise ValueError("training.lr_patience cannot be negative")
    if not 0 < float(training.get("lr_factor", 0.5)) < 1:
        raise ValueError("training.lr_factor must be between 0 and 1")
    if float(training.get("min_learning_rate", 0)) < 0:
        raise ValueError("training.min_learning_rate cannot be negative")
    if float(training.get("gradient_clip", 0)) < 0:
        raise ValueError("training.gradient_clip cannot be negative")


def resolve_project_root(config: dict[str, Any], config_path: Path) -> Path:
    configured = config.get("project_root")
    if configured:
        return expand_path(configured, config_path.parent)
    return Path(__file__).resolve().parents[2]


def resolve_output_root(config: dict[str, Any], config_path: Path) -> Path:
    configured = config["experiment"].get("output_root", "runs")
    return expand_path(configured, config_path.parent)


def _grid_values(grid: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys = sorted(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def expand_jobs(config: dict[str, Any], selected: set[str] | None = None) -> list[ResolvedJob]:
    base_training = deep_merge(DEFAULT_TRAINING, config.get("training", {}))
    base_seed = int(config["experiment"].get("seed", 777))
    repeats = int(config["experiment"].get("repeats", 1))
    global_grid = (config.get("sweep") or {}).get("grid", {})
    jobs: list[ResolvedJob] = []

    for model_entry in config["models"]:
        if model_entry.get("enabled", True) is False:
            continue
        model = str(model_entry["name"]).lower().strip()
        counts = model_entry["speakers"]
        if isinstance(counts, int):
            counts = [counts]
        model_training = deep_merge(base_training, model_entry.get("training", {}))
        model_grid = dict(global_grid)
        model_grid.update((model_entry.get("sweep") or {}).get("grid", {}))
        checkpoints = model_entry.get("initial_checkpoints", {})
        for speakers_value in counts:
            speakers = int(speakers_value)
            selector = f"{model}:{speakers}"
            if selected and model not in selected and selector not in selected:
                continue
            spec = get_model_spec(model, speakers)
            configured_checkpoint = checkpoints.get(str(speakers), model_entry.get("initial_checkpoint", "auto"))
            for sweep_values in _grid_values(model_grid):
                training = deepcopy(model_training)
                for key, value in sweep_values.items():
                    set_dotted({"training": training}, key, value)
                _validate_training(training)
                if model == "mossformer2" and training["precision"] == "fp16":
                    raise ValueError(
                        "MossFormer2 does not support numerically stable fp16 training; "
                        "use training.precision: bf16 or fp32"
                    )
                if spec.backend == "speechbrain_sepformer" and int(training["gradient_accumulation"]) != 1:
                    raise ValueError(
                        f"{model}/{speakers}spk uses SpeechBrain and requires training.gradient_accumulation: 1"
                    )
                for repeat in range(1, repeats + 1):
                    seed = base_seed + repeat - 1
                    identity = {
                        "model": model,
                        "speakers": speakers,
                        "repeat": repeat,
                        "seed": seed,
                        "backend": spec.backend,
                        "sample_rate": spec.sample_rate,
                        "training": training,
                        "initial_checkpoint": configured_checkpoint,
                        "sweep": sweep_values,
                    }
                    digest = stable_hash(identity)
                    lr = format(float(training["learning_rate"]), ".2g")
                    label = safe_slug(f"lr{lr}_bs{training['batch_size']}_seed{seed}__{digest}")
                    jobs.append(
                        ResolvedJob(
                            model=model,
                            speakers=speakers,
                            repeat=repeat,
                            seed=seed,
                            backend=spec.backend,
                            sample_rate=spec.sample_rate,
                            training=training,
                            initial_checkpoint_value=str(configured_checkpoint) if configured_checkpoint is not None else None,
                            sweep_values=sweep_values,
                            run_hash=digest,
                            run_label=label,
                        )
                    )
    if not jobs:
        raise ValueError("No jobs matched the configuration and selection")
    return jobs
