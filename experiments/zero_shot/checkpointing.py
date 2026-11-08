from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from utils import PROJECT_ROOT, safe_slug


class CheckpointNotFoundError(FileNotFoundError):
    """A model/speaker folder is absent from a multi-model checkpoint root."""


class InvalidCheckpointError(RuntimeError):
    """A checkpoint path exists but does not satisfy its adapter contract."""


def default_checkpoint_path(model: str, speakers: int) -> Path:
    key = (model.lower().strip(), int(speakers))
    paths = {
        ("mossformer2", 2): PROJECT_ROOT / "checkpoints" / "MossFormer2_SS_16K",
        ("mossformer2", 3): PROJECT_ROOT
        / "checkpoints"
        / "mossformer2-wsj0mix-3spk",
        ("convtasnet", 2): PROJECT_ROOT / "checkpoints" / "convtasnet_gln",
        ("dprnn", 2): PROJECT_ROOT
        / "checkpoints"
        / "dprnn-wham-2spk"
        / "pytorch_model.bin",
        ("sepformer", 2): PROJECT_ROOT / "checkpoints" / "sepformer-wsj02mix",
        ("sepformer", 3): PROJECT_ROOT / "checkpoints" / "sepformer-wsj03mix",
    }
    try:
        return paths[key].resolve()
    except KeyError as exc:
        raise ValueError(
            f"No checkpoint format is configured for {key[0]}/{key[1]}spk."
        ) from exc


def resolve_checkpoint_from_root(root: str | Path, model: str, speakers: int) -> Path:
    """Resolve ``root/model/Nspk`` with ``root/model`` as a flat fallback."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise CheckpointNotFoundError(f"Checkpoint root not found: {root_path}")

    speaker_path = root_path / model / f"{int(speakers)}spk"
    if speaker_path.exists():
        return validate_checkpoint(model, speakers, speaker_path)

    flat_path = root_path / model
    if flat_path.is_dir() and _has_direct_checkpoint_files(model, speakers, flat_path):
        return validate_checkpoint(model, speakers, flat_path)

    raise CheckpointNotFoundError(
        f"No checkpoint found for {model}/{speakers}spk. Expected: {speaker_path}"
    )


def validate_checkpoint(model: str, speakers: int, path: str | Path) -> Path:
    model = model.lower().strip()
    speakers = int(speakers)
    checkpoint = Path(path).expanduser().resolve()

    if model == "dprnn":
        checkpoint = _resolve_dprnn_file(checkpoint)
        if not checkpoint.is_file():
            raise InvalidCheckpointError(f"DPRNN checkpoint file not found: {checkpoint}")
        return checkpoint

    if not checkpoint.is_dir():
        raise InvalidCheckpointError(
            f"{model}/{speakers}spk checkpoint must be a directory: {checkpoint}"
        )

    if model == "mossformer2" and speakers == 2:
        _resolve_clearvoice_checkpoint_files(checkpoint)
        return checkpoint

    required = _required_relative_files(model, speakers)
    missing = [str(checkpoint / relative) for relative in required if not (checkpoint / relative).is_file()]
    if missing:
        raise InvalidCheckpointError(
            f"Invalid {model}/{speakers}spk checkpoint folder. Missing:\n  "
            + "\n  ".join(missing)
        )
    return checkpoint


def checkpoint_files(model: str, speakers: int, path: str | Path) -> List[Path]:
    checkpoint = validate_checkpoint(model, speakers, path)
    if checkpoint.is_file():
        return [checkpoint]

    if model.lower().strip() == "mossformer2" and int(speakers) == 2:
        files = _resolve_clearvoice_checkpoint_files(checkpoint)
        for name in ("config.yaml", "config.json"):
            optional_path = checkpoint / name
            if optional_path.is_file():
                files.append(optional_path)
        return sorted(set(files), key=lambda item: str(item))

    files = [checkpoint / relative for relative in _required_relative_files(model, speakers)]
    optional = {
        ("convtasnet", 2): ["trainer.json"],
    }.get((model.lower().strip(), int(speakers)), [])
    files.extend(checkpoint / relative for relative in optional if (checkpoint / relative).is_file())
    return sorted(set(files), key=lambda item: str(item))


def checkpoint_fingerprint(model: str, speakers: int, path: str | Path) -> str:
    """Content fingerprint used to keep result resumption checkpoint-safe."""
    checkpoint = validate_checkpoint(model, speakers, path)
    base = checkpoint if checkpoint.is_dir() else checkpoint.parent
    digest = hashlib.sha256()
    for file_path in checkpoint_files(model, speakers, checkpoint):
        relative = file_path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def result_model_name(model: str, checkpoint: str | Path | None, run_name: str | None) -> str:
    model = model.lower().strip()
    if checkpoint is None:
        return model
    tag = safe_slug(run_name or _default_run_name(Path(checkpoint)))
    if not tag:
        raise ValueError("Checkpoint --run-name resolves to an empty value.")
    return f"{model}__{tag}"


def _required_relative_files(model: str, speakers: int) -> List[str]:
    key = (model.lower().strip(), int(speakers))
    requirements = {
        ("mossformer2", 3): ["config.json", "pytorch_model.bin"],
        ("convtasnet", 2): ["best.pt.tar", "mdl.json"],
        ("sepformer", 2): [
            "hyperparams.yaml",
            "encoder.ckpt",
            "decoder.ckpt",
            "masknet.ckpt",
        ],
        ("sepformer", 3): [
            "hyperparams.yaml",
            "encoder.ckpt",
            "decoder.ckpt",
            "masknet.ckpt",
        ],
    }
    try:
        return requirements[key]
    except KeyError as exc:
        raise ValueError(
            f"No checkpoint folder contract is configured for {key[0]}/{key[1]}spk."
        ) from exc


def _resolve_dprnn_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        return path
    preferred = [
        path / "pytorch_model.bin",
        path / "model.pth",
        path / "best_model.pth",
        path / "best.pt",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    candidates = sorted(
        file_path
        for pattern in ("*.bin", "*.pth", "*.pt")
        for file_path in path.glob(pattern)
        if file_path.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            f"DPRNN folder contains multiple possible checkpoints: {path}\n  "
            + "\n  ".join(str(item) for item in candidates)
        )
    raise InvalidCheckpointError(
        f"No DPRNN .bin/.pth/.pt checkpoint found directly inside: {path}"
    )


def _has_direct_checkpoint_files(model: str, speakers: int, path: Path) -> bool:
    if model == "dprnn":
        return any(path.glob("*.bin")) or any(path.glob("*.pth")) or any(path.glob("*.pt"))
    if model == "mossformer2" and int(speakers) == 2:
        return (path / "last_best_checkpoint").is_file() or (path / "last_checkpoint").is_file()
    return any((path / relative).is_file() for relative in _required_relative_files(model, speakers))


def _resolve_clearvoice_checkpoint_files(path: Path) -> List[Path]:
    pointer = next(
        (
            candidate
            for candidate in (
                path / "last_best_checkpoint",
                path / "last_checkpoint",
            )
            if candidate.is_file()
        ),
        None,
    )
    if pointer is None:
        raise InvalidCheckpointError(
            "MossFormer2 2spk checkpoint requires a text pointer named "
            f"last_best_checkpoint or last_checkpoint: {path}"
        )
    model_name = pointer.read_text(encoding="utf-8").splitlines()[0].strip()
    if not model_name:
        raise InvalidCheckpointError(f"Empty ClearVoice checkpoint pointer: {pointer}")
    model_file = (path / model_name).resolve()
    try:
        model_file.relative_to(path.resolve())
    except ValueError as exc:
        raise InvalidCheckpointError(
            f"ClearVoice checkpoint pointer must stay inside its folder: {pointer} -> {model_name}"
        ) from exc
    if not model_file.is_file():
        raise InvalidCheckpointError(
            f"ClearVoice checkpoint target is missing: {pointer} -> {model_file}"
        )
    return [pointer, model_file]


def _default_run_name(path: Path) -> str:
    path = path.expanduser().resolve()
    folder = path.parent if path.is_file() else path
    if folder.name in {"2spk", "3spk"} and folder.parent.parent.name:
        return folder.parent.parent.name
    if folder.name in {"mossformer2", "convtasnet", "dprnn", "sepformer"}:
        return folder.parent.name
    return folder.name
