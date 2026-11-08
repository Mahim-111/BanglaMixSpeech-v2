from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parent.parent


class MossFormer2Standalone3Spk8kAdapter:
    """MossFormer2 WSJ0-3mix 8 kHz adapter using the official standalone code.

    Model alias used by the benchmark:
        mossformer3_8k

    Official checkpoint:
        alibabasglab/mossformer2-wsj0mix-3spk

    Why this adapter exists:
        The Hugging Face repo advertises AutoModel, but many Transformers
        installations do not register model_type='mossformer2'. The official
        MossFormer2_standalone example instead uses:

            from model.mossformer2 import Mossformer2Wrapper
            model = Mossformer2Wrapper.from_pretrained('alibabasglab/mossformer2-wsj0mix-3spk')

        This adapter follows that path.

    Evaluation policy:
        - Original BanglaMix files remain untouched at 16 kHz.
        - Mixtures are resampled in memory to 8 kHz before model inference.
        - Evaluator must use evaluation_sr='model' so references and mixture
          baseline are also resampled to 8 kHz before PIT and metrics.
        - CSV is stored under results_8k/mossformer3_8k/3spk/ by benchmark.py.
    """

    name = "mossformer3_8k"
    sample_rate = 8000
    default_batch_size = 1
    max_batch_size = 1
    force_evaluation_sr = "model"
    result_family = "results_8k"

    repo_id = "alibabasglab/mossformer2-wsj0mix-3spk"
    github_repo = "https://github.com/alibabasglab/MossFormer2.git"

    def __init__(
        self,
        speakers: int = 3,
        device: str = "cuda:0",
        verbose: bool = False,
        repo_id: str | None = None,
    ):
        if int(speakers) != 3:
            raise ValueError("mossformer3_8k is a 3-speaker-only checkpoint. Use --speaker 3.")

        self.speakers = 3
        self.verbose = bool(verbose)
        bundled_checkpoint = ROOT / "checkpoints" / "mossformer2-wsj0mix-3spk"
        self.repo_id = repo_id or os.environ.get(
            "MOSSFORMER3_8K_MODEL", str(bundled_checkpoint)
        )
        if not Path(self.repo_id).expanduser().exists():
            raise FileNotFoundError(
                f"Bundled MossFormer2 3-speaker checkpoint is missing: {self.repo_id}\n"
                "Run: python scripts/verify_bundle.py --full"
            )
        self.device = torch.device(device if (not str(device).startswith("cuda") or torch.cuda.is_available()) else "cpu")

        self.standalone_dir = self._resolve_standalone_code()
        self._add_standalone_to_path()

        try:
            from model.mossformer2 import Mossformer2Wrapper  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Could not import Mossformer2Wrapper from official MossFormer2_standalone code.\n"
                f"Expected path: {self.standalone_dir}\n"
                "Try removing repos/MossFormer2 and rerun so the adapter can clone it again."
            ) from exc

        # Newer huggingface_hub versions look only for model.safetensors when a
        # local directory is passed to PyTorchModelHubMixin. The upstream
        # snapshot contains pytorch_model.bin, so load that safe state dict
        # directly for a version-independent, offline path.
        try:
            local_model_dir = Path(self.repo_id).expanduser().resolve()
            if local_model_dir.is_dir():
                config = json.loads((local_model_dir / "config.json").read_text())
                # Upstream __init__ eagerly moves to the default GPU. Build on
                # CPU first so cuda:N and explicit CPU requests are respected,
                # then move once after loading the state dict.
                original_cuda_available = torch.cuda.is_available
                try:
                    torch.cuda.is_available = lambda: False
                    self.model = Mossformer2Wrapper(config=config)
                finally:
                    torch.cuda.is_available = original_cuda_available
                state = torch.load(
                    local_model_dir / "pytorch_model.bin",
                    map_location="cpu",
                    weights_only=True,
                )
                self.model.load_state_dict(state, strict=True)
            else:
                self.model = Mossformer2Wrapper.from_pretrained(self.repo_id)
        except Exception as exc:
            raise RuntimeError(
                "Could not load the bundled MossFormer2 3-speaker checkpoint via "
                "Mossformer2Wrapper.from_pretrained(...).\n"
                f"Repo id: {self.repo_id}\n"
                "Required packages: huggingface_hub, torch, soundfile, scipy.\n"
                "Install if needed:\n"
                "  pip install -U huggingface_hub scipy soundfile\n"
                "Verify the bundle with: python scripts/verify_bundle.py --full"
            ) from exc

        # Some versions already call .to(cuda) inside __init__, but we force the requested device.
        self.model.to(self.device)
        self.model.eval()

        model_sr = int(getattr(self.model, "sample_rate", self.sample_rate))
        model_spks = int(getattr(self.model, "num_spks", self.speakers))
        if model_sr != self.sample_rate:
            raise RuntimeError(f"{self.repo_id} reports sample_rate={model_sr}; expected 8000.")
        if model_spks != self.speakers:
            raise RuntimeError(f"{self.repo_id} reports num_spks={model_spks}; expected 3.")

        if self.verbose:
            print(
                f"[mossformer3_8k] repo={self.repo_id} | sample_rate={self.sample_rate} | "
                f"speakers={self.speakers} | device={self.device} | standalone={self.standalone_dir}"
            )

    def separate(self, mixture_path, output_dir=None) -> torch.Tensor:
        audio, sr = self._read_audio_mono(mixture_path)
        audio = self._resample_np(audio, orig_sr=sr, new_sr=self.sample_rate)
        audio = self._sanitize_audio(audio)

        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)  # (1, T)

        with torch.no_grad():
            est_source = self.model.forward(wav)

        estimates = self._normalize_model_output(est_source)
        return estimates.detach().cpu().float()

    def separate_batch(self, samples: List[Dict], output_dir=None) -> Tuple[Dict[str, torch.Tensor], float]:
        # Full-utterance inference uses variable lengths; do it per mixture but keep model loaded once.
        results: Dict[str, torch.Tensor] = {}
        total_infer = 0.0
        for sample in samples:
            start = time.perf_counter()
            results[sample["id"]] = self.separate(sample["mixture"], output_dir=output_dir)
            total_infer += time.perf_counter() - start
        return results, total_infer

    def separate_and_save(self, mixture_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        estimates = self.separate(mixture_path)
        paths = []
        for idx in range(estimates.shape[0]):
            path = output_dir / f"s{idx + 1}.wav"
            sf.write(str(path), estimates[idx].numpy(), self.sample_rate, subtype="PCM_16")
            paths.append(str(path))
        return paths

    def _resolve_standalone_code(self) -> Path:
        # Allow explicit path for offline/reproducible experiments.
        env_path = os.environ.get("MOSSFORMER2_STANDALONE_DIR")
        if env_path:
            path = Path(env_path).expanduser().resolve()
            if (path / "model" / "mossformer2.py").exists():
                return path
            raise RuntimeError(
                "MOSSFORMER2_STANDALONE_DIR is set but does not contain model/mossformer2.py:\n"
                f"  {path}"
            )

        repo_root = ROOT / "repos" / "MossFormer2"
        standalone = repo_root / "MossFormer2_standalone"
        if (standalone / "model" / "mossformer2.py").exists():
            return standalone.resolve()

        raise RuntimeError(
            "Bundled MossFormer2 standalone source is missing:\n"
            f"  {standalone}\n"
            "Run: python scripts/verify_bundle.py --full"
        )

    def _add_standalone_to_path(self) -> None:
        # The standalone code imports from `model.mossformer2`, so add MossFormer2_standalone.
        s = str(self.standalone_dir)
        if s not in sys.path:
            sys.path.insert(0, s)

    def _normalize_model_output(self, est_source: torch.Tensor) -> torch.Tensor:
        """Convert official output shape (1,T,3) to benchmark shape (3,T)."""
        if not torch.is_tensor(est_source):
            raise RuntimeError(f"MossFormer2 returned non-tensor output: {type(est_source)}")
        x = est_source.detach()
        if x.ndim == 3 and x.shape[0] == 1 and x.shape[-1] == self.speakers:
            x = x[0].transpose(0, 1)  # (S,T)
        elif x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == self.speakers:
            x = x[0]
        elif x.ndim == 2 and x.shape[0] == self.speakers:
            pass
        elif x.ndim == 2 and x.shape[1] == self.speakers:
            x = x.transpose(0, 1)
        else:
            raise RuntimeError(f"Unsupported MossFormer2 output shape: {tuple(x.shape)}")

        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)

        # Match the official standalone inference.py behavior, which peak-normalizes
        # each separated signal before writing wav. This keeps PESQ/STOI finite and
        # avoids clipping while keeping SI-SDR unchanged by scale.
        outs = []
        for s in range(self.speakers):
            y = x[s]
            peak = y.abs().max().clamp_min(1e-8)
            outs.append((y / peak * 0.99).unsqueeze(0))
        return torch.cat(outs, dim=0).contiguous()

    @staticmethod
    def _read_audio_mono(path) -> Tuple[np.ndarray, int]:
        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError(f"Empty audio file: {path}")
        return audio, int(sr)

    @staticmethod
    def _sanitize_audio(audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak * 0.99
        return audio.astype(np.float32, copy=False)

    @staticmethod
    def _resample_np(audio: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
        orig_sr = int(orig_sr)
        new_sr = int(new_sr)
        if orig_sr == new_sr:
            return np.asarray(audio, dtype=np.float32)
        try:
            from scipy.signal import resample_poly
        except Exception as exc:
            raise RuntimeError(
                "Resampling requires scipy for mossformer3_8k. Install it with:\n"
                "  pip install scipy"
            ) from exc
        import math
        g = math.gcd(orig_sr, new_sr)
        y = resample_poly(audio, new_sr // g, orig_sr // g)
        return np.asarray(y, dtype=np.float32)
