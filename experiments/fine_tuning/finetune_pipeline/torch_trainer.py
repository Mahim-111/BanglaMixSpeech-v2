from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .model_factory import export_model, load_trainable_model, normalize_estimates
from .util import atomic_write_json


class SeparationDataset:
    def __init__(self, manifest: Path, sample_rate: int, speakers: int, segment_seconds: float, training: bool):
        import torch

        self.torch = torch
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise RuntimeError(f"Empty manifest: {manifest}")
        self.sample_rate = sample_rate
        self.speakers = speakers
        self.segment_samples = max(1, round(segment_seconds * sample_rate))
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _read(path: str) -> tuple[np.ndarray, int]:
        import soundfile as sf

        audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)

    @staticmethod
    def _resample(audio: np.ndarray, old_rate: int, new_rate: int) -> np.ndarray:
        if old_rate == new_rate:
            return audio
        from scipy.signal import resample_poly

        divisor = math.gcd(old_rate, new_rate)
        return np.asarray(resample_poly(audio, new_rate // divisor, old_rate // divisor), dtype=np.float32)

    def _crop_or_pad(self, arrays: list[np.ndarray]) -> list[np.ndarray]:
        length = min(len(value) for value in arrays)
        arrays = [value[:length] for value in arrays]
        if length >= self.segment_samples:
            if self.training:
                start = random.randint(0, length - self.segment_samples)
            else:
                start = max(0, (length - self.segment_samples) // 2)
            return [value[start : start + self.segment_samples] for value in arrays]
        return [np.pad(value, (0, self.segment_samples - length)) for value in arrays]

    def __getitem__(self, index: int):
        row = self.rows[index]
        mixture, rate = self._read(row["mixture"])
        mixture = self._resample(mixture, rate, self.sample_rate)
        sources = []
        for path in row["sources"]:
            source, source_rate = self._read(path)
            sources.append(self._resample(source, source_rate, self.sample_rate))
        arrays = self._crop_or_pad([mixture, *sources])
        mixture_tensor = self.torch.from_numpy(np.ascontiguousarray(arrays[0])).float()
        sources_tensor = self.torch.from_numpy(np.stack(arrays[1:], axis=0)).float()
        return mixture_tensor, sources_tensor


def pit_si_sdr_loss(estimates, references, epsilon: float = 1.0e-8):
    import torch

    estimates = estimates - estimates.mean(dim=-1, keepdim=True)
    references = references - references.mean(dim=-1, keepdim=True)
    estimate_matrix = estimates.unsqueeze(2)
    reference_matrix = references.unsqueeze(1)
    projection = (
        (estimate_matrix * reference_matrix).sum(dim=-1, keepdim=True)
        * reference_matrix
        / (reference_matrix.square().sum(dim=-1, keepdim=True) + epsilon)
    )
    noise = estimate_matrix - projection
    pairwise = 10.0 * torch.log10(
        (projection.square().sum(dim=-1) + epsilon) / (noise.square().sum(dim=-1) + epsilon)
    )
    speaker_count = references.shape[1]
    permutation_scores = []
    for order in permutations(range(speaker_count)):
        permutation_scores.append(
            torch.stack([pairwise[:, estimate, reference] for estimate, reference in enumerate(order)], dim=1).mean(dim=1)
        )
    best = torch.stack(permutation_scores, dim=1).max(dim=1).values
    return -best.mean()


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _atomic_torch_save(value: Any, path: Path) -> None:
    import torch

    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _state_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value.detach().cpu() for key, value in state.items()}


def _make_scaler(enabled: bool):
    import torch

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _run_epoch(
    model,
    loader,
    device,
    speakers: int,
    optimizer,
    scaler,
    precision: str,
    accumulation: int,
    clip: float,
    epoch: int,
    total_epochs: int,
    initial_global_step: int,
    progress_label: str = "",
):
    import torch

    training = optimizer is not None
    stage = "train" if training else "valid"
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    total = 0.0
    count = 0
    optimizer_steps = 0
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    autocast_enabled = precision in {"fp16", "bf16"} and (device.type == "cuda" or precision == "bf16")
    prefix = f"{progress_label} " if progress_label else ""
    description = f"{prefix}epoch {epoch}/{total_epochs} {stage}"
    with tqdm(
        total=len(loader),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        mininterval=0.5,
        smoothing=0.1,
        file=sys.stdout,
        leave=True,
        disable=False,
    ) as progress:
        for batch_index, (mixture, references) in enumerate(loader, start=1):
            mixture = mixture.to(device, non_blocking=True)
            references = references.to(device, non_blocking=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                    raw = model(mixture)
                    estimates = normalize_estimates(raw, speakers=speakers, batch_size=mixture.shape[0])
                    length = min(estimates.shape[-1], references.shape[-1])
                    loss = pit_si_sdr_loss(estimates[..., :length], references[..., :length])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite SI-SDR loss at batch {batch_index}: {loss.item()}")
                if training:
                    scaler.scale(loss / accumulation).backward()
                    if batch_index % accumulation == 0 or batch_index == len(loader):
                        scaler.unscale_(optimizer)
                        if clip >= 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
            loss_value = float(loss.detach().cpu())
            total += loss_value
            count += 1
            postfix = {"loss": f"{total / count:.4f}"}
            if training:
                postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
                postfix["step"] = initial_global_step + optimizer_steps
            if device.type == "cuda":
                postfix["gpu"] = f"{torch.cuda.memory_allocated(device) / (1024 ** 3):.1f}G"
            progress.set_postfix(postfix, refresh=False)
            progress.update(1)
    if not count:
        raise RuntimeError("Dataloader produced no batches")
    return total / count, count, optimizer_steps


def train(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    speakers = int(config["speakers"])
    sample_rate = int(config["sample_rate"])
    seed = int(config["seed"])
    _seed_everything(seed)

    requested_device = str(config["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested_device} requested but torch.cuda.is_available() is false")
    device = torch.device(requested_device)
    precision = str(training["precision"])
    if device.type == "cpu" and precision == "fp16":
        precision = "fp32"

    bundle = load_trainable_model(config)
    model = bundle.model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training["lr_factor"]),
        patience=int(training["lr_patience"]),
        min_lr=float(training["min_learning_rate"]),
    )
    scaler = _make_scaler(enabled=precision == "fp16" and device.type == "cuda")

    train_dataset = SeparationDataset(
        Path(config["manifests"]["train"]), sample_rate, speakers, float(training["segment_seconds"]), True
    )
    valid_dataset = SeparationDataset(
        Path(config["manifests"]["valid"]),
        sample_rate,
        speakers,
        float(training.get("validation_segment_seconds", training["segment_seconds"])),
        False,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": device.type == "cuda",
        "generator": generator,
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    best_epoch = 0
    best_global_step = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    last_path = run_dir / "last.pt"
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_loss = float(checkpoint.get("best_loss", best_loss))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_global_step = int(checkpoint.get("best_global_step", 0))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        history = list(checkpoint.get("history", []))
        print(f"Resuming from epoch {start_epoch} (best validation loss {best_loss:.4f})", flush=True)

    print(
        f"Starting {config['model']}/{speakers}spk on {device} ({precision}) | "
        f"epochs={training['epochs']} | train={len(train_dataset)} samples/{len(train_loader)} batches | "
        f"valid={len(valid_dataset)} samples/{len(valid_loader)} batches | "
        f"accumulation={training['gradient_accumulation']}",
        flush=True,
    )
    started = time.time()
    total_epochs = int(training["epochs"])
    progress_label = f"{config['model']}/{speakers}spk"
    for epoch in range(start_epoch, total_epochs + 1):
        train_loss, train_batches, optimizer_steps = _run_epoch(
            model, train_loader, device, speakers, optimizer, scaler, precision,
            int(training["gradient_accumulation"]), float(training["gradient_clip"]),
            epoch, total_epochs, global_step, progress_label,
        )
        global_step += optimizer_steps
        valid_loss, valid_batches, _ = _run_epoch(
            model, valid_loader, device, speakers, None, scaler, precision, 1,
            float(training["gradient_clip"]), epoch, total_epochs, global_step, progress_label,
        )
        scheduler.step(valid_loss)
        improved = valid_loss < best_loss
        if improved:
            best_loss = valid_loss
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "global_step": global_step,
            "elapsed_seconds": time.time() - started,
            "improved": improved,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if improved:
            best_global_step = global_step
            export_model(
                bundle,
                run_dir,
                _state_to_cpu(model.state_dict()),
                epoch,
                global_step,
                {**training, "sample_rate": sample_rate, "speakers": speakers},
            )
        state = {
            "model": _state_to_cpu(model.state_dict()),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "best_global_step": best_global_step,
            "bad_epochs": bad_epochs,
            "history": history,
            "run_hash": config["run_hash"],
        }
        # Exactly one resumable checkpoint is retained and atomically replaced.
        _atomic_torch_save(state, last_path)
        if bad_epochs >= int(training["patience"]):
            print(f"Early stopping after {bad_epochs} epochs without improvement", flush=True)
            break

    export_dir = run_dir / "export"
    if not last_path.is_file() or not export_dir.is_dir() or not any(export_dir.iterdir()):
        raise RuntimeError("Training produced no last/best checkpoint pair")
    summary = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "best_global_step": best_global_step,
        "global_step": global_step,
        "last_checkpoint": str(last_path.resolve()),
        "best_checkpoint": str(export_dir.resolve()),
        "export_dir": str(export_dir.resolve()),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    atomic_write_json(run_dir / "training_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal PyTorch trainer for BanglaMix fine-tuning")
    parser.add_argument("--run-config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.run_config).read_text(encoding="utf-8"))
    train(config)


if __name__ == "__main__":
    main()
