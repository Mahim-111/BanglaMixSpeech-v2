from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple
import re

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_path(path: str | Path, name: str = "path") -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"{name} does not exist: {p}")
    return p


def safe_slug(text: str | None) -> str:
    if text is None or str(text).strip() == "":
        return "none"
    normalized = (
        str(text)
        .replace("%", "pct")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )
    return re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-") or "none"


def parse_cuda_index(device: str | torch.device | None) -> int:
    if device is None:
        return -1
    d = str(device).lower()
    if not d.startswith("cuda"):
        return -1
    if ":" in d:
        try:
            return int(d.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def wants_cuda(device: str | torch.device | None) -> bool:
    return str(device).lower().startswith("cuda")


def read_audio_mono(path: str | Path) -> Tuple[torch.Tensor, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    return torch.from_numpy(audio), int(sr)


def write_sources(output_dir: str | Path, estimates: torch.Tensor, sample_rate: int) -> List[str]:
    # Use soundfile instead of torchaudio.save to avoid TorchCodec/FFmpeg issues
    # in newer torchaudio builds. This path is only used when explicitly saving
    # separated wavs; the benchmark itself is CSV-only.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimates = normalize_sources_tensor(estimates)

    paths: List[str] = []
    for i in range(estimates.shape[0]):
        out = output_dir / f"s{i + 1}.wav"
        audio = estimates[i].detach().cpu().numpy().astype(np.float32)
        sf.write(str(out), audio, int(sample_rate))
        paths.append(str(out))
    return paths


def normalize_sources_tensor(x, expected_speakers: int | None = None) -> torch.Tensor:
    """Return separated audio as float tensor with shape (C, T).

    Handles common outputs: (C,T), (T,C), (1,T,C), (1,C,T), list of arrays.
    """
    if isinstance(x, torch.Tensor):
        t = x.detach()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        tensors = []
        for item in x:
            if isinstance(item, torch.Tensor):
                y = item.detach().cpu()
            else:
                y = torch.as_tensor(item)
            y = y.float()
            if y.ndim > 1:
                # Treat multi-channel source as mono.
                if y.shape[0] <= y.shape[-1]:
                    y = y[0]
                else:
                    y = y[:, 0]
            tensors.append(y)
        min_len = min(y.numel() for y in tensors)
        return torch.stack([y[:min_len] for y in tensors], dim=0).float().contiguous()
    else:
        t = torch.as_tensor(x)

    t = t.float().detach().cpu()

    # Remove leading singleton batch/channel dimensions when safe.
    while t.ndim > 2 and 1 in t.shape:
        # Prefer removing a batch dimension, not the speaker dimension.
        squeeze_dim = next(i for i, s in enumerate(t.shape) if s == 1)
        t = t.squeeze(squeeze_dim)

    if t.ndim == 1:
        if expected_speakers not in (None, 1):
            raise ValueError(
                f"Expected {expected_speakers} speakers, but model returned mono tensor {tuple(t.shape)}"
            )
        return t.unsqueeze(0).contiguous()

    if t.ndim != 2:
        raise ValueError(f"Could not normalize estimate tensor shape: {tuple(t.shape)}")

    if expected_speakers is not None:
        if t.shape[0] == expected_speakers:
            out = t
        elif t.shape[1] == expected_speakers:
            out = t.transpose(0, 1)
        else:
            # Heuristic fallback: smaller axis is usually speaker axis.
            out = t if t.shape[0] < t.shape[1] else t.transpose(0, 1)
    else:
        out = t if t.shape[0] < t.shape[1] else t.transpose(0, 1)

    if expected_speakers is not None:
        if out.shape[0] < expected_speakers:
            raise ValueError(
                f"Model returned {out.shape[0]} sources, expected {expected_speakers}. Shape={tuple(out.shape)}"
            )
        if out.shape[0] > expected_speakers:
            out = out[:expected_speakers]

    return out.contiguous()


def stack_reference_sources(source_paths: Iterable[str | Path]) -> Tuple[torch.Tensor, int]:
    waves = []
    sample_rate = None
    for src in source_paths:
        wav, sr = read_audio_mono(src)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(f"Reference sample-rate mismatch: expected {sample_rate}, got {sr} for {src}")
        waves.append(wav)

    if not waves:
        raise ValueError("No reference sources were provided.")

    min_len = min(w.numel() for w in waves)
    refs = torch.stack([w[:min_len] for w in waves], dim=0).float().contiguous()
    return refs, int(sample_rate)


def resample_sources(sources: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    sources = normalize_sources_tensor(sources)
    if int(orig_sr) == int(new_sr):
        return sources
    return AF.resample(sources.float().cpu(), orig_freq=int(orig_sr), new_freq=int(new_sr)).contiguous()


def align_lengths(*signals: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    min_len = min(int(x.shape[-1]) for x in signals)
    return tuple(x[..., :min_len].contiguous() for x in signals)
