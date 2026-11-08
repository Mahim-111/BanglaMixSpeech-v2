from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from .util import atomic_write_json, atomic_write_text


def _read_audio_at_sample_rate(path: str | Path, target_sample_rate: int):
    """Read mono audio and resample it to SepFormer's native rate."""
    import math

    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    target_sample_rate = int(target_sample_rate)
    if target_sample_rate <= 0:
        raise ValueError(f"target_sample_rate must be positive, got {target_sample_rate}")

    audio, source_sample_rate = sf.read(
        str(path), always_2d=False, dtype="float32"
    )
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1:
        raise RuntimeError(f"Expected mono audio at {path}, got shape {audio.shape}")
    if audio.size == 0:
        raise RuntimeError(f"Empty audio file: {path}")
    if not np.isfinite(audio).all():
        raise RuntimeError(f"Non-finite audio samples in {path}")

    source_sample_rate = int(source_sample_rate)
    if source_sample_rate != target_sample_rate:
        divisor = math.gcd(source_sample_rate, target_sample_rate)
        audio = resample_poly(
            audio,
            target_sample_rate // divisor,
            source_sample_rate // divisor,
        )

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    return torch.from_numpy(audio)


def _speechbrain_dataio_prep(hparams, sb):
    """Build SpeechBrain datasets with explicit model-rate audio loading.

    The upstream WSJ0Mix recipe assumes its WAV files are already at the rate
    declared by ``sample_rate``. BanglaMix is stored at 16 kHz, while the WSJ0
    SepFormer checkpoints are 8 kHz, so using the recipe's unmodified reader
    silently trains on the wrong time and frequency scale.
    """
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )
    datasets = [train_data, valid_data, test_data]
    target_sample_rate = int(hparams["sample_rate"])

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        return _read_audio_at_sample_rate(mix_wav, target_sample_rate)

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        return _read_audio_at_sample_rate(s1_wav, target_sample_rate)

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        return _read_audio_at_sample_rate(s2_wav, target_sample_rate)

    pipelines = [audio_pipeline_mix, audio_pipeline_s1, audio_pipeline_s2]
    output_keys = ["id", "mix_sig", "s1_sig", "s2_sig"]

    if int(hparams["num_spks"]) == 3:

        @sb.utils.data_pipeline.takes("s3_wav")
        @sb.utils.data_pipeline.provides("s3_sig")
        def audio_pipeline_s3(s3_wav):
            return _read_audio_at_sample_rate(s3_wav, target_sample_rate)

        pipelines.append(audio_pipeline_s3)
        output_keys.append("s3_sig")

    for pipeline in pipelines:
        sb.dataio.dataset.add_dynamic_item(datasets, pipeline)
    sb.dataio.dataset.set_output_keys(datasets, output_keys)
    return train_data, valid_data, test_data


def _is_validation_checkpoint(checkpoint) -> bool:
    """Return whether a checkpoint may compete for best validation loss."""
    return (
        checkpoint.path.name != "CKPT+last"
        and checkpoint.meta.get("kind") != "last"
    )


@contextmanager
def _best_only_retention(checkpointer):
    """Make a recipe's save-and-prune call retain only its validation best.

    SpeechBrain's ``save_and_keep_only`` keeps the most recent checkpoint by
    default in addition to checkpoints selected by ``min_keys``. A separately
    maintained ``CKPT+last`` must also be excluded from pruning; otherwise it
    can win a tie for best and then be overwritten, silently deleting the true
    historical best checkpoint.
    """
    original = checkpointer.save_and_keep_only

    def save_validation_best(*args, **kwargs):
        existing_predicate = kwargs.get("ckpt_predicate")

        def candidate(checkpoint):
            if not _is_validation_checkpoint(checkpoint):
                return False
            return existing_predicate is None or existing_predicate(checkpoint)

        # Pass a fresh empty list explicitly. Some SpeechBrain versions use a
        # mutable default for importance_keys and append the recency key to it.
        kwargs["keep_recent"] = False
        kwargs["importance_keys"] = []
        kwargs["ckpt_predicate"] = candidate
        return original(*args, **kwargs)

    checkpointer.save_and_keep_only = save_validation_best
    try:
        yield
    finally:
        checkpointer.save_and_keep_only = original


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def _replace_top_level(text: str, replacements: dict[str, Any]) -> str:
    remaining = dict(replacements)
    output = []
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):(?:\s.*)?$", line)
        if match and match.group(1) in remaining:
            key = match.group(1)
            output.append(f"{key}: {_yaml_scalar(remaining.pop(key))}")
        else:
            output.append(line)
    if remaining:
        missing = ", ".join(sorted(remaining))
        raise RuntimeError(f"SpeechBrain template is missing expected top-level keys: {missing}")
    return "\n".join(output) + "\n"


def render_hparams(config: dict[str, Any]) -> Path:
    project_root = Path(config["project_root"])
    run_dir = Path(config["run_dir"])
    recipe = project_root / "repos" / "speechbrain" / "recipes" / "WSJ0Mix" / "separation"
    template = recipe / "hparams" / "sepformer.yaml"
    if not template.is_file():
        raise FileNotFoundError(f"SpeechBrain SepFormer template not found: {template}")
    training = config["training"]
    output_folder = run_dir / "speechbrain"
    save_folder = output_folder / "save"
    replacements = {
        "seed": int(config["seed"]),
        "data_folder": str(Path(config["dataset_root"])),
        "experiment_name": config["run_hash"],
        "output_folder": str(output_folder),
        "train_log": str(output_folder / "train_log.txt"),
        "save_folder": str(save_folder),
        "train_data": config["manifests"]["train_speechbrain"],
        "valid_data": config["manifests"]["valid_speechbrain"],
        "test_data": config["manifests"]["test_speechbrain"],
        "skip_prep": True,
        "precision": training["precision"],
        "num_spks": int(config["speakers"]),
        "noprogressbar": bool(training.get("no_progress", False)),
        "save_audio": False,
        "sample_rate": int(config["sample_rate"]),
        "N_epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "lr": float(training["learning_rate"]),
        "clip_grad_norm": float(training["gradient_clip"]),
        "limit_training_signal_len": True,
        "training_signal_len": round(float(training["segment_seconds"]) * int(config["sample_rate"])),
        "dynamic_mixing": False,
        "use_wavedrop": bool(training.get("use_wavedrop", False)),
        "use_speedperturb": bool(training.get("use_speed_perturb", False)),
        "use_rand_shift": False,
    }
    rendered = _replace_top_level(template.read_text(encoding="utf-8"), replacements)
    rendered, worker_replacements = re.subn(
        r"(?m)^(\s+num_workers:)\s*\d+\s*$",
        rf"\g<1> {int(training['num_workers'])}",
        rendered,
        count=1,
    )
    if worker_replacements != 1:
        raise RuntimeError("Could not set SpeechBrain dataloader num_workers")
    rendered = re.sub(
        r"(?m)^(\s+weight_decay:)\s*[^#\n]+",
        rf"\g<1> {float(training['weight_decay'])}",
        rendered,
        count=1,
    )
    rendered = re.sub(
        r"(?m)^(\s+factor:)\s*[^#\n]+",
        rf"\g<1> {float(training['lr_factor'])}",
        rendered,
        count=1,
    )
    rendered = re.sub(
        r"(?m)^(\s+patience:)\s*[^#\n]+",
        rf"\g<1> {int(training['lr_patience'])}",
        rendered,
        count=1,
    )
    initial = Path(config["initial_checkpoint"])
    collect_in = run_dir / "pretrained"
    rendered += f"""

# Added by BanglaMix_FineTune_Pipeline. Local-only initialization keeps runs reproducible.
pretrained_separator: !new:speechbrain.utils.parameter_transfer.Pretrainer
    collect_in: {_yaml_scalar(str(collect_in))}
    loadables:
        encoder: !ref <Encoder>
        decoder: !ref <Decoder>
        masknet: !ref <MaskNet>
    paths:
        encoder: {_yaml_scalar(str(initial / 'encoder.ckpt'))}
        decoder: {_yaml_scalar(str(initial / 'decoder.ckpt'))}
        masknet: {_yaml_scalar(str(initial / 'masknet.ckpt'))}
"""
    target = run_dir / "resolved_speechbrain_hparams.yaml"
    atomic_write_text(target, rendered)
    return target


def _load_recipe_module(recipe_dir: Path):
    path = recipe_dir / "train.py"
    spec = importlib.util.spec_from_file_location("banglamix_speechbrain_recipe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SpeechBrain recipe: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _best_checkpoint(save_folder: Path) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for folder in save_folder.glob("CKPT*"):
        if folder.name == "CKPT+last":
            continue
        metadata_path = folder / "CKPT.yaml"
        if not metadata_path.is_file():
            continue
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        metric = metadata.get("si-snr", float("inf"))
        candidates.append((float(metric), folder, metadata))
    if not candidates:
        raise RuntimeError(f"SpeechBrain produced no checkpoint under {save_folder}")
    _, folder, metadata = min(candidates, key=lambda value: value[0])
    return folder, metadata


def _export(config: dict[str, Any], checkpoint: Path) -> Path:
    run_dir = Path(config["run_dir"])
    initial = Path(config["initial_checkpoint"])
    output = run_dir / "export"
    staging = run_dir / ".best_export.tmp"
    previous = run_dir / ".best_export.previous"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in ("encoder.ckpt", "decoder.ckpt", "masknet.ckpt"):
        source = checkpoint / name
        if not source.is_file():
            raise RuntimeError(f"SpeechBrain checkpoint is missing {name}: {checkpoint}")
        # Hard-link into a staging folder: this consumes no additional model
        # storage and lets the complete export directory be swapped atomically.
        os.link(source, staging / name)
    # Architecture changes are intentionally not exposed by this pipeline, so
    # the original inference hyperparameters remain the correct loader contract.
    shutil.copy2(initial / "hyperparams.yaml", staging / "hyperparams.yaml")
    if previous.exists():
        shutil.rmtree(previous)
    if output.exists():
        output.replace(previous)
    staging.replace(output)
    if previous.exists():
        shutil.rmtree(previous)
    # Removing the source checkpoint leaves the hard-linked export as the sole
    # best weight set on disk.
    shutil.rmtree(checkpoint)
    return output


def train(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(config["project_root"])
    speechbrain_root = project_root / "repos" / "speechbrain"
    recipe_dir = speechbrain_root / "recipes" / "WSJ0Mix" / "separation"
    sys.path.insert(0, str(speechbrain_root))
    sys.path.insert(0, str(recipe_dir))
    hparams_path = render_hparams(config)
    started = time.time()

    try:
        import speechbrain as sb
        from hyperpyyaml import load_hyperpyyaml
        from speechbrain.utils.distributed import run_on_main
        from speechbrain.utils.logger import get_logger
    except ImportError as exc:
        raise RuntimeError("SepFormer fine-tuning requires SpeechBrain and hyperpyyaml") from exc

    recipe = _load_recipe_module(recipe_dir)
    recipe.logger = get_logger("banglamix_finetune.sepformer")

    class RetentionSeparation(recipe.Separation):
        """Keep SpeechBrain's single best plus one fixed-name last checkpoint."""

        def on_stage_end(self, stage, stage_loss, epoch):
            # The upstream recipe calls save_and_keep_only during validation.
            # Narrow that call to ordinary validation checkpoints so CKPT+last
            # remains independently resumable and cannot replace the best.
            with _best_only_retention(self.checkpointer):
                super().on_stage_end(stage, stage_loss, epoch)
            if stage == sb.Stage.VALID:
                self.checkpointer.save_checkpoint(
                    meta={"si-snr": stage_loss, "kind": "last"},
                    name="last",
                )
    with hparams_path.open(encoding="utf-8") as handle:
        hparams = load_hyperpyyaml(handle)

    requested_device = str(config["device"])
    run_opts = {"device": requested_device}
    if requested_device == "cpu" and hparams.get("precision") == "fp16":
        hparams["precision"] = "bf16"
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=str(hparams_path),
        overrides={},
    )
    train_data, valid_data, _ = _speechbrain_dataio_prep(hparams, sb)
    run_on_main(hparams["pretrained_separator"].collect_files)
    hparams["pretrained_separator"].load_collected()

    separator = RetentionSeparation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    separator.fit(
        separator.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )
    checkpoint, metadata = _best_checkpoint(Path(hparams["save_folder"]))
    export_dir = _export(config, checkpoint)
    summary = {
        "status": "completed",
        "best_valid_loss": float(metadata.get("si-snr", float("nan"))),
        "last_checkpoint": str((Path(hparams["save_folder"]) / "CKPT+last").resolve()),
        "best_checkpoint": str(export_dir.resolve()),
        "export_dir": str(export_dir.resolve()),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(Path(config["run_dir"]) / "training_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal SpeechBrain SepFormer trainer")
    parser.add_argument("--run-config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.run_config).read_text(encoding="utf-8"))
    train(config)


if __name__ == "__main__":
    main()
