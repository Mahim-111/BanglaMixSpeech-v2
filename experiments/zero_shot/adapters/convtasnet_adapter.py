from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
import torch

from utils import PROJECT_ROOT, normalize_sources_tensor, parse_cuda_index, read_audio_mono, resample_sources, safe_slug, write_sources


class ConvTasNetAdapter:
    """Original funcwj ConvTasNet repo adapter.

    The bundled pretrain/gln.tgz checkpoint uses the repo config with
    fs = 8000 and WSJ0-2mix 8 kHz data. BanglaMix is 16 kHz, so this
    adapter writes temporary 8 kHz mixture wavs before calling the
    official separate.py script. Returned estimates are 8 kHz; the
    evaluator can resample them back to the reference rate for metrics.
    """

    name = "convtasnet"
    sample_rate = 8000
    checkpoint_sample_rate = 8000
    # The original repo writes separated wavs for every SCP entry.
    # Large SCPs can look frozen and can pressure RAM/disk, so keep chunks safe.
    default_batch_size = 64
    max_batch_size = 128

    def __init__(
        self,
        speakers: int = 2,
        device: str = "cuda:0",
        checkpoint: str | Path = "checkpoints/convtasnet_gln",
        verbose: bool = False,
    ):
        if speakers != 2:
            raise ValueError("The configured original ConvTasNet GLN checkpoint is 2-speaker only.")

        self.speakers = int(speakers)
        self.device = device
        self.gpu = parse_cuda_index(device)
        self.verbose = bool(verbose)
        self.repo = PROJECT_ROOT / "repos" / "ConvTasNet"
        self.script = self.repo / "nnet" / "separate.py"
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        self.checkpoint = checkpoint_path.resolve()

        for name, path in {
            "ConvTasNet repo": self.repo,
            "ConvTasNet separate.py": self.script,
            "ConvTasNet checkpoint dir": self.checkpoint,
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")

    def separate(self, mixture_path, output_dir=None):
        mixture_path = Path(mixture_path).expanduser().resolve()
        sample = {"id": mixture_path.stem, "mixture": str(mixture_path)}
        result, _elapsed = self.separate_batch([sample], output_dir=output_dir)
        return result[sample["id"]]

    def separate_batch(self, samples: Iterable[dict], output_dir=None) -> Tuple[Dict[str, torch.Tensor], float]:
        samples = list(samples)
        if not samples:
            return {}, 0.0

        if output_dir is None:
            with tempfile.TemporaryDirectory(prefix="convtasnet_batch_") as tmpdir:
                return self._separate_batch_to_dir(samples, Path(tmpdir))

        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._separate_batch_to_dir(samples, output_dir)

    def _separate_batch_to_dir(self, samples: List[dict], output_dir: Path) -> Tuple[Dict[str, torch.Tensor], float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        scp = output_dir / "mix.scp"
        input_dir = output_dir / "_input_8k"
        input_dir.mkdir(parents=True, exist_ok=True)

        key_to_sample_id: Dict[str, str] = {}
        lines: List[str] = []
        for idx, sample in enumerate(samples):
            mixture_path = Path(sample["mixture"]).expanduser().resolve()
            if not mixture_path.exists():
                raise FileNotFoundError(f"Mixture not found: {mixture_path}")

            key = f"{idx:06d}__{safe_slug(sample['id'])}"
            key_to_sample_id[key] = sample["id"]

            # The official funcwj ConvTasNet WaveReader checks file SR against
            # --fs and does not resample internally. Therefore every BanglaMix
            # 16 kHz mixture must be converted to a temporary 8 kHz wav first.
            conv_input = self._prepare_input_wav(mixture_path, input_dir / f"{key}.wav")
            lines.append(f"{key} {conv_input}\n")

        scp.write_text("".join(lines), encoding="utf-8")

        cmd = [
            sys.executable,
            str(self.script),
            str(self.checkpoint),
            "--input",
            str(scp),
            "--gpu",
            str(self.gpu),
            "--fs",
            str(self.sample_rate),
            "--dump-dir",
            str(output_dir),
        ]

        start = time.perf_counter()
        self._run_subprocess(cmd)
        elapsed = time.perf_counter() - start

        wav_files = sorted(output_dir.rglob("*.wav"))
        estimates_by_id: Dict[str, torch.Tensor] = {}
        missing: List[str] = []

        for key, sample_id in key_to_sample_id.items():
            files = self._find_outputs_for_key(output_dir, wav_files, key)
            if len(files) < self.speakers:
                missing.append(sample_id)
                continue
            estimates_by_id[sample_id] = self._load_sources(files[: self.speakers])

        if missing:
            preview = [str(p.relative_to(output_dir)) for p in wav_files[:30]]
            raise RuntimeError(
                "ConvTasNet inference finished, but outputs could not be matched for "
                f"{len(missing)} sample(s). First missing ids: {missing[:5]}\n"
                f"Output dir: {output_dir}\n"
                f"First wav files found: {preview}\n"
                "Expected a layout like spk1/<scp-key>.wav and spk2/<scp-key>.wav."
            )

        return estimates_by_id, elapsed


    def _prepare_input_wav(self, mixture_path: Path, out_path: Path) -> Path:
        """Create an 8 kHz mono temporary input wav for the official script."""
        wav, sr = read_audio_mono(mixture_path)
        wav = wav.float().cpu()

        if int(sr) != self.sample_rate:
            wav = resample_sources(wav.unsqueeze(0), orig_sr=int(sr), new_sr=self.sample_rate)[0]

        audio = wav.detach().cpu().numpy().astype(np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # PCM_16 is safest for the original repo's wav reader and keeps temp files small.
        sf.write(str(out_path), audio, self.sample_rate, subtype="PCM_16")
        return out_path

    def _run_subprocess(self, cmd: List[str]) -> None:
        if self.verbose:
            subprocess.run(cmd, cwd=str(self.repo / "nnet"), check=True)
            return

        proc = subprocess.run(
            cmd,
            cwd=str(self.repo / "nnet"),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-80:])
            raise RuntimeError(f"ConvTasNet inference failed. Last output lines:\n{tail}")

    def _find_outputs_for_key(self, output_dir: Path, wav_files: List[Path], key: str) -> List[Path]:
        # Official layout: dump-dir/spk1/<key>.wav, dump-dir/spk2/<key>.wav
        expected = [output_dir / f"spk{i}" / f"{key}.wav" for i in range(1, self.speakers + 1)]
        if all(p.exists() for p in expected):
            return expected

        candidates: List[Path] = []
        for p in wav_files:
            rel_parts = p.relative_to(output_dir).parts
            if p.stem == key or p.stem.startswith(key) or key in rel_parts:
                candidates.append(p)

        candidates = sorted(set(candidates), key=self._speaker_sort_key)
        return candidates[: self.speakers]

    def _load_sources(self, wav_paths: List[Path]) -> torch.Tensor:
        waves = []
        for wav_path in wav_paths:
            audio, sr = sf.read(str(wav_path), always_2d=False)
            if int(sr) != self.sample_rate:
                raise RuntimeError(
                    f"ConvTasNet output SR mismatch: expected {self.sample_rate}, got {sr} from {wav_path}"
                )
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            waves.append(np.asarray(audio, dtype=np.float32))

        min_len = min(len(w) for w in waves)
        estimates = torch.from_numpy(np.stack([w[:min_len] for w in waves], axis=0)).float()
        return normalize_sources_tensor(estimates, expected_speakers=self.speakers)

    @staticmethod
    def _speaker_sort_key(path: Path):
        text = "/".join(path.parts).lower()
        patterns = [
            r"(?:^|[/_\-])spk\s*([0-9]+)(?:$|[/_\-.])",
            r"(?:^|[/_\-])speaker\s*([0-9]+)(?:$|[/_\-.])",
            r"(?:^|[/_\-])src\s*([0-9]+)(?:$|[/_\-.])",
            r"(?:^|[/_\-])s([0-9]+)(?:$|[/_\-.])",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return (int(m.group(1)), str(path))
        return (999, str(path))

    def separate_and_save(self, mixture_path, output_dir):
        estimates = self.separate(mixture_path)
        return write_sources(output_dir, estimates, self.sample_rate)
