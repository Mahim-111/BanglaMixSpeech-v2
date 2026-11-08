from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


@dataclass
class ModelBundle:
    model: Any
    export_kind: str
    export_metadata: dict[str, Any]


def _torch_load(path: Path, map_location: str = "cpu"):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Checkpoint does not contain a state dict: {type(payload)}")


def _load_flexible(model, state: dict[str, Any]) -> None:
    own = model.state_dict()
    normalized = {}
    for key, value in state.items():
        candidates = (key, key.removeprefix("module."), f"module.{key}")
        selected = next((candidate for candidate in candidates if candidate in own and own[candidate].shape == value.shape), None)
        if selected is not None:
            normalized[selected] = value
    missing_before = len(own) - len(normalized)
    if missing_before:
        missing_preview = [key for key in own if key not in normalized][:12]
        raise RuntimeError(
            f"Initial checkpoint is architecture-incompatible: {missing_before} parameter(s) did not match. "
            f"First missing: {missing_preview}"
        )
    model.load_state_dict(normalized, strict=True)


def _resolve_clearvoice_file(folder: Path) -> Path:
    for pointer_name in ("last_best_checkpoint", "last_checkpoint"):
        pointer = folder / pointer_name
        if pointer.is_file():
            lines = pointer.read_text(encoding="utf-8").splitlines()
            target = lines[0].strip() if lines else ""
            path = (folder / target).resolve()
            try:
                path.relative_to(folder.resolve())
            except ValueError as exc:
                raise RuntimeError(f"Checkpoint pointer escapes its folder: {pointer}") from exc
            if path.is_file():
                return path
    raise FileNotFoundError(f"No valid ClearVoice checkpoint pointer in {folder}")


def _load_clearvoice(project_root: Path, initial: Path, speakers: int, sample_rate: int) -> ModelBundle:
    train_dir = project_root / "repos" / "ClearerVoice-Studio" / "train" / "speech_separation"
    if not (train_dir / "networks.py").is_file():
        raise FileNotFoundError(f"ClearVoice training source not found: {train_dir}")
    sys.path.insert(0, str(train_dir))
    network_module = importlib.import_module("networks")
    args = SimpleNamespace(
        network="MossFormer2_SS_16K" if sample_rate == 16000 else "MossFormer2_SS_8K",
        num_spks=speakers,
        encoder_kernel_size=16,
        encoder_embedding_dim=512,
        mossformer_sequence_dim=512,
        num_mossformer_layer=24,
    )
    model = network_module.network_wrapper(args).ss_network
    payload = _torch_load(_resolve_clearvoice_file(initial))
    _load_flexible(model, _state_dict(payload))
    return ModelBundle(
        model=model,
        export_kind="clearvoice",
        export_metadata={"network": args.network, **vars(args)},
    )


def _load_mossformer_standalone(project_root: Path, initial: Path) -> ModelBundle:
    import torch

    standalone = project_root / "repos" / "MossFormer2" / "MossFormer2_standalone"
    if not (standalone / "model" / "mossformer2.py").is_file():
        raise FileNotFoundError(f"MossFormer2 standalone source not found: {standalone}")
    sys.path.insert(0, str(standalone))
    module = importlib.import_module("model.mossformer2")
    config = json.loads((initial / "config.json").read_text(encoding="utf-8"))
    original = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False
        model = module.Mossformer2Wrapper(config=config)
    finally:
        torch.cuda.is_available = original
    state = _torch_load(initial / "pytorch_model.bin")
    model.load_state_dict(_state_dict(state), strict=True)
    return ModelBundle(model=model, export_kind="mossformer_standalone", export_metadata={"config": config})


def _load_convtasnet(project_root: Path, initial: Path) -> ModelBundle:
    nnet_dir = project_root / "repos" / "ConvTasNet" / "nnet"
    if not (nnet_dir / "conv_tas_net.py").is_file():
        raise FileNotFoundError(f"ConvTasNet source not found: {nnet_dir}")
    sys.path.insert(0, str(nnet_dir))
    module = importlib.import_module("conv_tas_net")
    model_config = json.loads((initial / "mdl.json").read_text(encoding="utf-8"))
    model = module.ConvTasNet(**model_config)
    payload = _torch_load(initial / "best.pt.tar")
    model.load_state_dict(_state_dict(payload), strict=True)
    return ModelBundle(model=model, export_kind="convtasnet", export_metadata={"model_config": model_config})


def _load_dprnn(initial: Path) -> ModelBundle:
    try:
        from asteroid.models import DPRNNTasNet
    except ImportError as exc:
        raise RuntimeError("DPRNN fine-tuning requires the asteroid package") from exc
    import torch

    checkpoint = initial / "pytorch_model.bin"
    original_load = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    try:
        torch.load = compatible_load
        model = DPRNNTasNet.from_pretrained(str(checkpoint))
    finally:
        torch.load = original_load
    template = _torch_load(checkpoint)
    return ModelBundle(model=model, export_kind="dprnn", export_metadata={"template": template})


def load_trainable_model(config: dict[str, Any]) -> ModelBundle:
    project_root = Path(config["project_root"])
    initial = Path(config["initial_checkpoint"])
    backend = config["backend"]
    speakers = int(config["speakers"])
    sample_rate = int(config["sample_rate"])
    if backend == "torch_mossformer2_clearvoice":
        return _load_clearvoice(project_root, initial, speakers, sample_rate)
    if backend == "torch_mossformer2_standalone":
        return _load_mossformer_standalone(project_root, initial)
    if backend == "torch_convtasnet":
        return _load_convtasnet(project_root, initial)
    if backend == "torch_dprnn":
        return _load_dprnn(initial)
    raise ValueError(f"Unsupported torch backend: {backend}")


def normalize_estimates(output, speakers: int, batch_size: int):
    import torch

    if isinstance(output, (list, tuple)):
        if len(output) != speakers:
            raise RuntimeError(f"Model returned {len(output)} outputs for {speakers} speakers")
        output = torch.stack(list(output), dim=1)
    if torch.is_tensor(output) and output.ndim == 2 and batch_size == 1:
        if output.shape[0] == speakers:
            output = output.unsqueeze(0)
        elif output.shape[1] == speakers:
            output = output.transpose(0, 1).unsqueeze(0)
    if not torch.is_tensor(output) or output.ndim != 3:
        raise RuntimeError(f"Expected a rank-3 model output, got {type(output)} / {getattr(output, 'shape', None)}")
    if output.shape[0] == batch_size and output.shape[1] == speakers:
        estimates = output
    elif output.shape[0] == batch_size and output.shape[-1] == speakers:
        estimates = output.transpose(1, 2)
    elif output.shape[0] == speakers and output.shape[1] == batch_size:
        estimates = output.permute(1, 0, 2)
    else:
        raise RuntimeError(f"Cannot normalize model output shape {tuple(output.shape)} for B={batch_size}, S={speakers}")
    return estimates.float()


def export_model(
    bundle: ModelBundle,
    run_dir: Path,
    model_state: dict[str, Any],
    epoch: int,
    global_step: int,
    training: dict[str, Any],
) -> Path:
    import torch
    import yaml

    output = run_dir / "export"
    output.mkdir(parents=True, exist_ok=True)
    if bundle.export_kind == "clearvoice":
        filename = "last_best_checkpoint.pt"
        _atomic_torch_save(
            {"model": model_state, "epoch": epoch, "step": global_step},
            output / filename,
        )
        (output / "last_best_checkpoint").write_text(filename + "\n", encoding="utf-8")
        config = dict(bundle.export_metadata)
        config.update({"sampling_rate": training.get("sample_rate"), "num_spks": training.get("speakers")})
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    elif bundle.export_kind == "mossformer_standalone":
        _atomic_torch_save(model_state, output / "pytorch_model.bin")
        (output / "config.json").write_text(
            json.dumps(bundle.export_metadata["config"], indent=2) + "\n", encoding="utf-8"
        )
    elif bundle.export_kind == "convtasnet":
        _atomic_torch_save(
            {"epoch": epoch, "model_state_dict": model_state},
            output / "best.pt.tar",
        )
        (output / "mdl.json").write_text(
            json.dumps(bundle.export_metadata["model_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        trainer = {
            "optimizer": "adam",
            "optimizer_kwargs": {"lr": training["learning_rate"], "weight_decay": training["weight_decay"]},
            "factor": training["lr_factor"],
            "patience": training["lr_patience"],
            "min_lr": training["min_learning_rate"],
        }
        (output / "trainer.json").write_text(json.dumps(trainer, indent=2) + "\n", encoding="utf-8")
    elif bundle.export_kind == "dprnn":
        payload = dict(bundle.export_metadata["template"])
        payload["state_dict"] = model_state
        infos = dict(payload.get("infos", {}))
        infos["banglamix_finetune"] = {"epoch": epoch, "global_step": global_step, "training": training}
        payload["infos"] = infos
        _atomic_torch_save(payload, output / "pytorch_model.bin")
    else:
        raise ValueError(f"Unsupported export kind: {bundle.export_kind}")
    return output


def _atomic_torch_save(value: Any, path: Path) -> None:
    import torch

    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)
