from __future__ import annotations

from pathlib import Path


MODEL_NAMES = ("mossformer2", "convtasnet", "dprnn", "sepformer")


def supported_speakers(model_name: str) -> tuple[int, ...]:
    name = model_name.lower().strip()
    if name in {"mossformer2", "sepformer"}:
        return (2, 3)
    if name in {"convtasnet", "dprnn"}:
        return (2,)
    raise ValueError(f"Unsupported model: {model_name}")


def load_model(
    model_name: str,
    speakers: int = 2,
    device: str = "cuda:0",
    verbose: bool = False,
    checkpoint: str | Path | None = None,
):
    """Load one bundled zero-shot model.

    The public checkpoints are not uniform across speaker counts. MossFormer2
    therefore uses the official 16 kHz ClearVoice checkpoint for two speakers
    and the official 8 kHz standalone WSJ0-3mix checkpoint for three speakers.
    """
    name = model_name.lower().strip()
    speakers = int(speakers)

    if speakers not in supported_speakers(name):
        allowed = ", ".join(str(value) for value in supported_speakers(name))
        raise ValueError(f"{name} supports speaker count(s): {allowed}; got {speakers}.")

    if name == "mossformer2":
        if speakers == 2:
            from adapters.clearvoice_adapter import ClearVoiceAdapter

            kwargs = {}
            if checkpoint is not None:
                kwargs["checkpoint_dir"] = checkpoint
            return ClearVoiceAdapter(
                speakers=2, device=device, verbose=verbose, **kwargs
            )

        from adapters.mossformer2_standalone_adapter import (
            MossFormer2Standalone3Spk8kAdapter,
        )

        return MossFormer2Standalone3Spk8kAdapter(
            speakers=3,
            device=device,
            verbose=verbose,
            repo_id=str(checkpoint) if checkpoint is not None else None,
        )

    if name == "convtasnet":
        from adapters.convtasnet_adapter import ConvTasNetAdapter

        kwargs = {}
        if checkpoint is not None:
            kwargs["checkpoint"] = checkpoint
        return ConvTasNetAdapter(speakers=2, device=device, verbose=verbose, **kwargs)

    if name == "dprnn":
        from adapters.dprnn_adapter import DPRNNAdapter

        return DPRNNAdapter(
            speakers=2,
            device=device,
            verbose=verbose,
            pretrained=str(checkpoint) if checkpoint is not None else None,
        )

    if name == "sepformer":
        from adapters.speechbrain_adapter import SpeechBrainAdapter

        return SpeechBrainAdapter(
            speakers=speakers,
            device=device,
            verbose=verbose,
            checkpoint_dir=checkpoint,
        )

    raise ValueError(f"Unsupported model: {model_name}")
