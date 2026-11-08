from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModelSpec:
    name: str
    speakers: int
    sample_rate: int
    backend: str
    required_checkpoint_files: tuple[str, ...]
    notes: str


MODEL_SPECS: dict[tuple[str, int], ModelSpec] = {
    ("mossformer2", 2): ModelSpec(
        "mossformer2", 2, 16000, "torch_mossformer2_clearvoice",
        ("last_best_checkpoint",),
        "ClearVoice MossFormer2 16 kHz; exports ClearVoice pointer format.",
    ),
    ("mossformer2", 3): ModelSpec(
        "mossformer2", 3, 8000, "torch_mossformer2_standalone",
        ("config.json", "pytorch_model.bin"),
        "Official standalone WSJ0-3mix architecture; exports Hugging Face state dict format.",
    ),
    ("convtasnet", 2): ModelSpec(
        "convtasnet", 2, 8000, "torch_convtasnet",
        ("best.pt.tar", "mdl.json"),
        "funcwj Conv-TasNet GLN architecture used by the benchmark adapter.",
    ),
    ("dprnn", 2): ModelSpec(
        "dprnn", 2, 8000, "torch_dprnn",
        ("pytorch_model.bin",),
        "Asteroid DPRNNTasNet architecture used by the benchmark adapter.",
    ),
    ("sepformer", 2): ModelSpec(
        "sepformer", 2, 8000, "speechbrain_sepformer",
        ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"),
        "SpeechBrain SepFormer WSJ0-2mix architecture.",
    ),
    ("sepformer", 3): ModelSpec(
        "sepformer", 3, 8000, "speechbrain_sepformer",
        ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"),
        "SpeechBrain SepFormer WSJ0-3mix architecture.",
    ),
}


def get_model_spec(model: str, speakers: int) -> ModelSpec:
    key = (model.lower().strip(), int(speakers))
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        supported = ", ".join(f"{name}/{count}spk" for name, count in MODEL_SPECS)
        raise ValueError(f"Unsupported model/speaker pair {key[0]}/{key[1]}spk. Supported: {supported}") from exc


def model_names() -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for name, _ in MODEL_SPECS))


def supported_speakers(model: str) -> tuple[int, ...]:
    name = model.lower().strip()
    values = tuple(count for candidate, count in MODEL_SPECS if candidate == name)
    if not values:
        raise ValueError(f"Unsupported model: {model}")
    return values


def _checkpoint_candidates(project_root: Path, model: str, speakers: int) -> Iterable[Path]:
    portable = project_root / "BanglaMix_ZeroShot_Benchmark" / "checkpoints"
    root = project_root / "checkpoints"
    mapping = {
        ("mossformer2", 2): [root / "MossFormer2_SS_16K", portable / "MossFormer2_SS_16K"],
        ("mossformer2", 3): [portable / "mossformer2-wsj0mix-3spk", root / "mossformer2-wsj0mix-3spk"],
        ("convtasnet", 2): [root / "convtasnet_gln", portable / "convtasnet_gln"],
        ("dprnn", 2): [portable / "dprnn-wham-2spk", root / "dprnn-wham-2spk"],
        ("sepformer", 2): [root / "sepformer-wsj02mix", portable / "sepformer-wsj02mix"],
        ("sepformer", 3): [root / "sepformer-wsj03mix", portable / "sepformer-wsj03mix"],
    }
    return mapping[(model, speakers)]


def resolve_initial_checkpoint(
    project_root: Path,
    model: str,
    speakers: int,
    configured: str | Path | None,
    config_dir: Path,
) -> Path:
    spec = get_model_spec(model, speakers)
    if configured and str(configured).lower() != "auto":
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = config_dir / path
        candidates = [path.resolve()]
    else:
        candidates = list(_checkpoint_candidates(project_root, spec.name, spec.speakers))

    failures: list[str] = []
    for candidate in candidates:
        if not candidate.is_dir():
            failures.append(f"not a directory: {candidate}")
            continue
        missing = [name for name in spec.required_checkpoint_files if not (candidate / name).is_file()]
        if missing:
            failures.append(f"{candidate} missing {', '.join(missing)}")
            continue
        if spec.backend == "torch_mossformer2_clearvoice":
            pointer = (candidate / "last_best_checkpoint").read_text(encoding="utf-8").splitlines()
            target = pointer[0].strip() if pointer else ""
            if not target or not (candidate / target).is_file():
                failures.append(f"invalid last_best_checkpoint pointer: {candidate}")
                continue
        return candidate.resolve()

    details = "\n  ".join(failures) if failures else "no candidates"
    raise FileNotFoundError(
        f"No compatible initial checkpoint for {spec.name}/{spec.speakers}spk. Checked:\n  {details}"
    )

