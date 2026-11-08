from __future__ import annotations

from pathlib import Path

import torch
import torchaudio.functional as AF

from utils import PROJECT_ROOT, normalize_sources_tensor, read_audio_mono, write_sources


class DPRNNAdapter:
    name = "dprnn"
    sample_rate = 8000

    def __init__(
        self,
        speakers: int = 2,
        device: str = "cuda:0",
        pretrained: str | None = None,
        verbose: bool = False,
    ):
        if speakers != 2:
            raise ValueError("The configured Asteroid DPRNN checkpoint is 2-speaker only.")

        self.speakers = speakers
        self.device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
        bundled = PROJECT_ROOT / "checkpoints" / "dprnn-wham-2spk" / "pytorch_model.bin"
        selected = Path(pretrained).expanduser() if pretrained else bundled
        if selected.is_dir():
            preferred = [
                selected / "pytorch_model.bin",
                selected / "model.pth",
                selected / "best_model.pth",
                selected / "best.pt",
            ]
            selected = next((path for path in preferred if path.is_file()), selected)
        self.pretrained = str(selected.resolve())
        if not Path(self.pretrained).is_file():
            raise FileNotFoundError(
                "Bundled DPRNN checkpoint is missing: " + self.pretrained + "\n"
                "Run: python scripts/verify_bundle.py --full"
            )
        self.verbose = bool(verbose)

        from asteroid.models import DPRNNTasNet

        # PyTorch 2.6+ changed torch.load default to weights_only=True.
        # Some Asteroid checkpoints still require weights_only=False.
        original_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        try:
            torch.load = patched_load
            self.model = DPRNNTasNet.from_pretrained(self.pretrained)
        finally:
            torch.load = original_load

        self.model.to(self.device)
        self.model.eval()

    def separate(self, mixture_path, output_dir=None):
        # Do not use torchaudio.load here.
        # Newer torchaudio versions route WAV decoding through TorchCodec,
        # which can fail when PyTorch/TorchCodec/FFmpeg versions are mismatched.
        # soundfile is stable for PCM WAV and is already used by the evaluator.
        wav, sr = read_audio_mono(mixture_path)  # (T,), CPU float32
        wav = wav.float().unsqueeze(0)           # Asteroid expects (B, T)

        if sr != self.sample_rate:
            wav = AF.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        wav = wav.to(self.device)
        with torch.no_grad():
            estimates = self.model(wav)

        # Common Asteroid shape: (B, C, T). For B=1 -> (C,T).
        estimates = normalize_sources_tensor(estimates, expected_speakers=self.speakers)
        return estimates

    def separate_and_save(self, mixture_path, output_dir):
        estimates = self.separate(mixture_path)
        return write_sources(output_dir, estimates, self.sample_rate)
