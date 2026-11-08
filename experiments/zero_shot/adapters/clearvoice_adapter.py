from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parent.parent


class ClearVoiceAdapter:
    """ClearVoice-Studio MossFormer2 adapter.

    Contract used by the benchmark:
      separate(mixture_path, output_dir=None) -> torch.Tensor, shape (C, T)
      separate_batch(samples, output_dir=None) -> (dict[id -> Tensor(C,T)], inference_sec)

    Important:
      - MossFormer2_SS_16K public ClearVoice checkpoint is 2-speaker, 16 kHz.
      - For 3-speaker 16 kHz, provide a verified ClearVoice-compatible checkpoint
        with a last_best_checkpoint pointer and its referenced model file using either:
          export MOSSFORMER_3SPK_CHECKPOINT_DIR=/absolute/path/to/checkpoint_dir
        or:
          export MOSSFORMER_3SPK_HF_REPO=<hf_repo_id_of_clearvoice_3spk_16k_checkpoint>

    The public alibabasglab/mossformer2-wsj0mix-3spk checkpoint is NOT used here
    by default because its config reports sample_rate=8000 and it is not the same
    ClearVoice inference-checkpoint format expected by inference.py.
    """

    sample_rate = 16000

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints/MossFormer2_SS_16K",
        device: str = "cuda:0",
        speakers: int = 2,
        verbose: bool = False,
    ):
        self.device = str(device)
        self.speakers = int(speakers)
        self.verbose = bool(verbose)

        if self.speakers not in (2, 3):
            raise ValueError("ClearVoiceAdapter currently supports only 2 or 3 speakers.")

        # Conservative default chunks. 3-speaker is heavier and creates more temp wavs.
        self.default_batch_size = 16 if self.speakers == 2 else 4
        self.max_recommended_batch_size = 64 if self.speakers == 2 else 16

        self.repo = ROOT / "repos" / "ClearerVoice-Studio" / "train" / "speech_separation"
        self.inference = self.repo / "inference.py"
        self.config = self.repo / "config" / "inference" / "MossFormer2_SS_16K.yaml"

        if self.speakers == 2:
            self.checkpoint = self._resolve_2spk_checkpoint(checkpoint_dir)
        else:
            self.checkpoint = self._resolve_3spk_checkpoint()

        self._validate_runtime_paths()

    # ------------------------------------------------------------------
    # Checkpoint resolution
    # ------------------------------------------------------------------

    def _resolve_2spk_checkpoint(self, checkpoint_dir: str | Path) -> Path:
        """Resolve/download the official 2-speaker ClearVoice 16 kHz checkpoint."""
        path = Path(checkpoint_dir)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()

        if self._is_clearvoice_checkpoint_dir(path):
            return path

        # Auto-download official ClearVoice-format 2-speaker 16 kHz checkpoint.
        # Hugging Face repo has a pointer and the model file it names.
        repo_id = os.environ.get(
            "MOSSFORMER_2SPK_HF_REPO",
            "alibabasglab/MossFormer2_SS_16K",
        ).strip()
        if self.verbose:
            print(f"[MossFormer] 2-speaker checkpoint missing at {path}")
            print(f"[MossFormer] downloading {repo_id} -> {path}")
        downloaded = self._download_hf_checkpoint(repo_id, path)
        if not self._is_clearvoice_checkpoint_dir(downloaded):
            raise RuntimeError(
                "Downloaded 2-speaker MossFormer checkpoint is not ClearVoice-compatible.\n"
                f"Repo          : {repo_id}\n"
                f"Downloaded to : {downloaded}\n"
                "Expected files: last_best_checkpoint (text pointer) and its model file"
            )
        return downloaded

    def _resolve_3spk_checkpoint(self) -> Path:
        """Resolve a 3-speaker 16 kHz ClearVoice-compatible checkpoint.

        Priority:
          1. MOSSFORMER_3SPK_CHECKPOINT_DIR
          2. checkpoints/MossFormer2_SS_16K_3SPK
          3. MOSSFORMER_3SPK_HF_REPO, downloaded and validated

        We intentionally do NOT auto-download alibabasglab/mossformer2-wsj0mix-3spk
        as a 16 kHz ClearVoice checkpoint because its config says sample_rate=8000.
        """
        env_dir = os.environ.get("MOSSFORMER_3SPK_CHECKPOINT_DIR", "").strip()
        candidates: List[Path] = []
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        candidates.append(ROOT / "checkpoints" / "MossFormer2_SS_16K_3SPK")

        for path in candidates:
            path = path.resolve()
            if self._is_clearvoice_checkpoint_dir(path):
                self._validate_optional_hf_config(path, expected_sr=16000, expected_spks=3)
                return path

        hf_repo = os.environ.get("MOSSFORMER_3SPK_HF_REPO", "").strip()
        if hf_repo:
            local_dir = (ROOT / "checkpoints" / self._safe_repo_name(hf_repo)).resolve()
            if self.verbose:
                print(f"[MossFormer] downloading 3-speaker checkpoint {hf_repo} -> {local_dir}")
            downloaded = self._download_hf_checkpoint(hf_repo, local_dir)
            if not self._is_clearvoice_checkpoint_dir(downloaded):
                raise RuntimeError(
                    "Downloaded MossFormer 3-speaker checkpoint is not ClearVoice-compatible.\n"
                    f"Repo          : {hf_repo}\n"
                    f"Downloaded to : {downloaded}\n"
                    "Expected files: last_best_checkpoint (text pointer) and its model file\n\n"
                    "Important: the public alibabasglab/mossformer2-wsj0mix-3spk repo is a "
                    "Transformers-style 8 kHz WSJ0-3mix checkpoint with encoder.ckpt, "
                    "decoder.ckpt, masknet.ckpt, and pytorch_model.bin. It is not the same "
                    "ClearVoice inference.py checkpoint format and cannot be used by this "
                    "ClearVoice adapter as a 16 kHz BanglaMix baseline."
                )
            self._validate_optional_hf_config(downloaded, expected_sr=16000, expected_spks=3)
            return downloaded

        raise RuntimeError(
            "No public ClearVoice-format 3-speaker 16 kHz MossFormer checkpoint is configured.\n\n"
            "This adapter can auto-download, but it needs a repo id that actually contains "
            "a ClearVoice-compatible 3-speaker 16 kHz checkpoint containing:\n"
            "  last_best_checkpoint (text pointer) and its referenced model file\n\n"
            "Set it like this:\n"
            "  export MOSSFORMER_3SPK_HF_REPO=<repo_id_of_ClearVoice_3spk_16k_checkpoint>\n"
            "or place the checkpoint here:\n"
            f"  {ROOT / 'checkpoints' / 'MossFormer2_SS_16K_3SPK'}\n\n"
            "I did not default to alibabasglab/mossformer2-wsj0mix-3spk because that public "
            "checkpoint is 3-speaker but 8 kHz and Transformers-style, not ClearVoice "
            "MossFormer2_SS_16K inference.py format."
        )

    @staticmethod
    def _safe_repo_name(repo_id: str) -> str:
        return repo_id.replace("/", "__").replace(":", "_")

    @staticmethod
    def _is_clearvoice_checkpoint_dir(path: Path) -> bool:
        if not path.is_dir():
            return False
        for pointer_name in ("last_best_checkpoint", "last_checkpoint"):
            pointer = path / pointer_name
            if not pointer.is_file():
                continue
            try:
                model_name = pointer.read_text(encoding="utf-8").splitlines()[0].strip()
            except Exception:
                continue
            if model_name and (path / model_name).is_file():
                return True
        return False

    @staticmethod
    def _clearvoice_checkpoint_files(path: Path) -> Tuple[Path | None, Path | None]:
        for pointer_name in ("last_best_checkpoint", "last_checkpoint"):
            pointer = path / pointer_name
            if not pointer.is_file():
                continue
            model_name = pointer.read_text(encoding="utf-8").splitlines()[0].strip()
            if model_name:
                return pointer, path / model_name
        return None, None

    def _download_hf_checkpoint(self, repo_id: str, local_dir: Path) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "huggingface_hub is required for MOSSFORMER_3SPK_HF_REPO downloads.\n"
                "Install it with: pip install huggingface_hub"
            ) from exc

        local_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return local_dir.resolve()

    def _validate_optional_hf_config(self, path: Path, expected_sr: int, expected_spks: int) -> None:
        cfg_path = path / "config.json"
        if not cfg_path.exists():
            return
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            return

        sr = cfg.get("sample_rate") or cfg.get("sampling_rate")
        spks = cfg.get("num_spks") or cfg.get("masknet_numspks") or cfg.get("num_speakers")

        if sr is not None and int(sr) != int(expected_sr):
            raise RuntimeError(
                f"MossFormer checkpoint at {path} reports sample_rate={sr}, "
                f"but this benchmark run requires {expected_sr} Hz."
            )
        if spks is not None and int(spks) != int(expected_spks):
            raise RuntimeError(
                f"MossFormer checkpoint at {path} reports num_spks={spks}, "
                f"but this benchmark run requires {expected_spks} speakers."
            )

    def _validate_runtime_paths(self) -> None:
        missing = []
        for label, path in [
            ("ClearVoice repo", self.repo),
            ("ClearVoice inference.py", self.inference),
            ("ClearVoice config", self.config),
            ("MossFormer checkpoint dir", self.checkpoint),
        ]:
            if not path.exists():
                missing.append(f"{label}: {path}")
        pointer, model_file = self._clearvoice_checkpoint_files(self.checkpoint)
        if pointer is None or not pointer.is_file():
            missing.append(
                f"MossFormer checkpoint pointer: {self.checkpoint}/last_best_checkpoint "
                "or last_checkpoint"
            )
        if model_file is None or not model_file.is_file():
            missing.append(f"MossFormer checkpoint target: {model_file}")
        if missing:
            raise FileNotFoundError("Missing MossFormer/ClearVoice path(s):\n" + "\n".join(missing))

    # ------------------------------------------------------------------
    # Public separation API
    # ------------------------------------------------------------------

    def separate(self, mixture_path, output_dir=None) -> torch.Tensor:
        sample = {"id": Path(mixture_path).stem, "mixture": str(mixture_path)}
        estimates_by_id, _ = self.separate_batch([sample], output_dir=output_dir)
        return estimates_by_id[sample["id"]]

    def separate_batch(self, samples: List[dict], output_dir=None) -> Tuple[Dict[str, torch.Tensor], float]:
        if not samples:
            return {}, 0.0

        # Hard safety: do not let 3-speaker runs create huge temp folders.
        if len(samples) > self.max_recommended_batch_size:
            merged: Dict[str, torch.Tensor] = {}
            total_time = 0.0
            for start in range(0, len(samples), self.max_recommended_batch_size):
                sub = samples[start:start + self.max_recommended_batch_size]
                got, sec = self._separate_batch_to_dir(sub, output_dir=output_dir)
                merged.update(got)
                total_time += sec
            return merged, total_time

        return self._separate_batch_to_dir(samples, output_dir=output_dir)

    # ------------------------------------------------------------------
    # ClearVoice subprocess implementation
    # ------------------------------------------------------------------

    def _separate_batch_to_dir(self, samples: List[dict], output_dir=None) -> Tuple[Dict[str, torch.Tensor], float]:
        temp_parent = Path(output_dir).resolve() if output_dir is not None else None
        if temp_parent is not None:
            temp_parent.mkdir(parents=True, exist_ok=True)
            work_dir = Path(tempfile.mkdtemp(prefix="banglamix_mossformer_chunk_", dir=str(temp_parent)))
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="banglamix_mossformer_chunk_"))

        input_dir = work_dir / "input"
        cv_output_dir = work_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        cv_output_dir.mkdir(parents=True, exist_ok=True)

        id_to_key: Dict[str, str] = {}
        try:
            for idx, sample in enumerate(samples):
                sample_id = str(sample["id"])
                key = f"{idx:06d}__{sample_id}"
                id_to_key[sample_id] = key
                src = Path(sample["mixture"]).resolve()
                dst = input_dir / f"{key}.wav"
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

            cmd = [
                sys.executable,
                str(self.inference),
                "--config",
                str(self.config),
                "--checkpoint-dir",
                str(self.checkpoint),
                "--input-path",
                str(input_dir),
                "--output-dir",
                str(cv_output_dir),
                "--num-spks",
                str(self.speakers),
                "--use-cuda",
                "1" if self.device.startswith("cuda") else "0",
            ]

            t0 = time.time()
            self._run_subprocess(cmd)
            inference_sec = time.time() - t0

            estimates_by_id: Dict[str, torch.Tensor] = {}
            for sample_id, key in id_to_key.items():
                waves = []
                for spk in range(1, self.speakers + 1):
                    wav_path = cv_output_dir / f"{key}_s{spk}.wav"
                    if not wav_path.exists():
                        found = sorted(p.name for p in cv_output_dir.glob(f"{key}*.wav"))
                        raise RuntimeError(
                            f"Missing MossFormer output for sample={sample_id}, speaker={spk}.\n"
                            f"Expected: {wav_path}\n"
                            f"Found matching files: {found}"
                        )
                    audio, sr = sf.read(str(wav_path), always_2d=False)
                    if sr != self.sample_rate:
                        raise RuntimeError(
                            f"MossFormer output SR mismatch: expected {self.sample_rate}, got {sr} from {wav_path}"
                        )
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    waves.append(audio.astype(np.float32, copy=False))

                min_len = min(len(w) for w in waves)
                estimates_by_id[sample_id] = torch.tensor(
                    np.stack([w[:min_len] for w in waves]),
                    dtype=torch.float32,
                )

            return estimates_by_id, inference_sec
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_subprocess(self, cmd: List[str]) -> None:
        # The benchmark has a top-level utils.py while ClearVoice has its own
        # utils/ namespace package. Do not leak the benchmark PYTHONPATH into
        # this upstream subprocess, or Python can resolve the wrong module.
        child_env = os.environ.copy()
        child_env.pop("PYTHONPATH", None)
        if self.verbose:
            subprocess.run(cmd, cwd=str(self.repo), env=child_env, check=True)
            return

        proc = subprocess.run(
            cmd,
            cwd=str(self.repo),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            lines = (proc.stdout or "").splitlines()
            tail = "\n".join(lines[-40:])
            raise RuntimeError(
                "MossFormer/ClearVoice inference failed. Last output lines:\n"
                f"{tail}"
            )
