from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from finetune_pipeline.config import expand_jobs, validate_config
from finetune_pipeline.manifests import prepare_manifests
from finetune_pipeline.model_factory import ModelBundle, export_model, normalize_estimates
from finetune_pipeline.orchestrator import _run_and_tee
from finetune_pipeline.speechbrain_trainer import (
    _best_only_retention,
    _export as export_speechbrain,
    _is_validation_checkpoint,
    _read_audio_at_sample_rate,
)
from finetune_pipeline.registry import get_model_spec, resolve_initial_checkpoint
from finetune_pipeline.torch_trainer import _make_scaler, _run_epoch


def base_config(dataset_root: str) -> dict:
    return {
        "version": 1,
        "experiment": {"name": "unit", "seed": 10, "repeats": 2},
        "dataset": {"root": dataset_root},
        "training": {
            "epochs": 2,
            "batch_size": 1,
            "learning_rate": 1e-4,
            "segment_seconds": 1,
            "num_workers": 0,
            "gradient_accumulation": 1,
            "precision": "fp32",
        },
        "models": [{"name": "convtasnet", "speakers": [2]}],
        "sweep": {"grid": {"training.learning_rate": [1e-4, 2e-4]}},
    }


class ConfigTests(unittest.TestCase):
    def test_grid_and_repeats_create_unique_jobs(self):
        config = base_config("/dataset")
        validate_config(config)
        jobs = expand_jobs(config)
        self.assertEqual(len(jobs), 4)
        self.assertEqual({job.seed for job in jobs}, {10, 11})
        self.assertEqual({job.training["learning_rate"] for job in jobs}, {1e-4, 2e-4})
        self.assertEqual(len({job.run_hash for job in jobs}), 4)

    def test_model_selector_supports_speaker_pair(self):
        config = base_config("/dataset")
        config["models"].append({"name": "sepformer", "speakers": [2, 3]})
        jobs = expand_jobs(config, selected={"sepformer:3"})
        self.assertTrue(jobs)
        self.assertEqual({(job.model, job.speakers) for job in jobs}, {("sepformer", 3)})

    def test_model_specific_grid_overrides_global_key(self):
        config = base_config("/dataset")
        config["experiment"]["repeats"] = 1
        config["models"][0]["sweep"] = {"grid": {"training.learning_rate": [3e-4]}}
        validate_config(config)
        jobs = expand_jobs(config)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].training["learning_rate"], 3e-4)

    def test_registry_matrix(self):
        self.assertEqual(get_model_spec("mossformer2", 2).sample_rate, 16000)
        self.assertEqual(get_model_spec("mossformer2", 3).sample_rate, 8000)
        with self.assertRaises(ValueError):
            get_model_spec("dprnn", 3)

    def test_mossformer2_rejects_unstable_fp16_training(self):
        config = base_config("/dataset")
        config["models"] = [{"name": "mossformer2", "speakers": [2]}]
        config["training"]["precision"] = "fp16"
        validate_config(config)
        with self.assertRaisesRegex(ValueError, "MossFormer2.*bf16 or fp32"):
            expand_jobs(config)

    def test_single_batch_time_speaker_output_is_normalized(self):
        import torch

        output = torch.zeros(100, 2)
        normalized = normalize_estimates(output, speakers=2, batch_size=1)
        self.assertEqual(tuple(normalized.shape), (1, 2, 100))


class ManifestTests(unittest.TestCase):
    def _sample(self, root: Path, speakers: int, split: str, variation: str, name: str) -> None:
        if speakers == 2:
            folder = root / "Mix_Speech" / "Mix_For_Two_Speaker" / split / variation / name
        else:
            folder = root / "Mix_Speech" / "Mix_For_Three_Speaker" / variation / split / name
        folder.mkdir(parents=True)
        (folder / "mix.wav").write_bytes(b"mix")
        for index in range(1, speakers + 1):
            (folder / f"s{index}.wav").write_bytes(f"s{index}".encode())

    def test_two_speaker_manifest_and_balancing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            for split in ("train", "validation", "test"):
                self._sample(root, 2, split, "overlap_25%", "a")
                self._sample(root, 2, split, "overlap_100%", "b")
            self._sample(root, 2, "train", "overlap_100%", "c")
            summary = prepare_manifests(
                {"root": str(root), "variations": "all", "verify_audio": "none", "balance_variations": True},
                2,
                Path(temporary) / "manifests",
            )
            self.assertEqual(summary["original_train_count"], 3)
            self.assertEqual(summary["outputs"]["train"]["count"], 4)
            rows = [json.loads(line) for line in Path(summary["outputs"]["train"]["jsonl"]).read_text().splitlines()]
            self.assertEqual(sum(bool(row.get("oversampled")) for row in rows), 1)
            header = Path(summary["outputs"]["train"]["speechbrain_csv"]).read_text().splitlines()[0]
            self.assertIn("s2_wav", header)

    def test_three_speaker_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            for split in ("train", "validation", "test"):
                self._sample(root, 3, split, "max_overlap_jitter12pct", "a")
            summary = prepare_manifests(
                {"root": str(root), "variations": "all", "verify_audio": "none"},
                3,
                Path(temporary) / "manifests",
            )
            header = Path(summary["outputs"]["valid"]["speechbrain_csv"]).read_text().splitlines()[0]
            self.assertIn("s3_wav", header)
            self.assertEqual(summary["outputs"]["test"]["count"], 1)

    def test_sepformer_reader_resamples_16k_audio_to_8k(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temporary:
            wav_path = Path(temporary) / "input_16k.wav"
            samples = np.arange(16000, dtype=np.float32)
            tone = 0.2 * np.sin(2.0 * np.pi * 440.0 * samples / 16000.0)
            sf.write(wav_path, tone, 16000, subtype="FLOAT")

            converted = _read_audio_at_sample_rate(wav_path, 8000)

            self.assertEqual(converted.dtype.is_floating_point, True)
            self.assertEqual(converted.ndim, 1)
            self.assertEqual(converted.numel(), 8000)
            self.assertTrue(converted.isfinite().all().item())

    def test_all_variations_are_discovered_per_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            self._sample(root, 2, "train", "uniform_15_100", "a")
            self._sample(root, 2, "validation", "uniform_15_100", "b")
            self._sample(root, 2, "test", "overlap_85_100", "c")
            summary = prepare_manifests(
                {"root": str(root), "variations": "all", "verify_audio": "none"},
                2,
                Path(temporary) / "manifests",
            )
            self.assertEqual(summary["settings"]["variations"]["train"], ["uniform_15_100"])
            self.assertEqual(summary["settings"]["variations"]["test"], ["overlap_85_100"])

    def test_cached_manifest_paths_are_rebased_after_move(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            for split in ("train", "validation", "test"):
                self._sample(root, 2, split, "uniform_15_100", "a")
            dataset = {"root": str(root), "variations": "all", "verify_audio": "none"}
            old_root = Path(temporary) / "old" / "manifests"
            summary = prepare_manifests(dataset, 2, old_root)

            new_root = Path(temporary) / "new" / "manifests"
            moved_folder = new_root / "2spk" / summary["fingerprint"]
            moved_folder.parent.mkdir(parents=True)
            shutil.move(summary["folder"], moved_folder)

            rebased = prepare_manifests(dataset, 2, new_root)
            self.assertEqual(Path(rebased["folder"]), moved_folder.resolve())
            self.assertEqual(Path(rebased["outputs"]["train"]["jsonl"]), moved_folder / "train.jsonl")
            persisted = json.loads((moved_folder / "manifest_summary.json").read_text())
            self.assertEqual(persisted["outputs"]["valid"]["scp_dir"], str((moved_folder / "scp" / "valid").resolve()))


class ProgressTests(unittest.TestCase):
    def test_torch_epoch_reports_live_progress_fields(self):
        import torch

        class TinySeparator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, mixture):
                return torch.stack((mixture * self.scale, -mixture * self.scale), dim=1)

        mixture = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        references = torch.stack((mixture, -mixture), dim=1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            loss, batches, steps = _run_epoch(
                TinySeparator(),
                [(mixture, references)],
                torch.device("cpu"),
                speakers=2,
                optimizer=None,
                scaler=_make_scaler(False),
                precision="fp32",
                accumulation=1,
                clip=5.0,
                epoch=1,
                total_epochs=2,
                initial_global_step=7,
            )
        self.assertTrue(math.isfinite(loss))
        self.assertEqual((batches, steps), (1, 0))
        self.assertIn("epoch 1/2 valid", output.getvalue())
        self.assertIn("loss=", output.getvalue())

    def test_subprocess_tee_forwards_carriage_return_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "train.log"
            output = io.StringIO()
            command = [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('batch 1\\r'); sys.stdout.flush(); "
                "sys.stdout.write('batch 2\\n'); sys.stdout.flush()",
            ]
            with contextlib.redirect_stdout(output):
                status = _run_and_tee(command, Path(temporary), log_path)
            self.assertEqual(status, 0)
            self.assertIn("batch 1\rbatch 2\n", output.getvalue())
            self.assertIn(b"batch 1\rbatch 2\n", log_path.read_bytes())


class CheckpointTests(unittest.TestCase):
    def test_speechbrain_best_retention_excludes_last_and_recency(self):
        class Checkpoint:
            def __init__(self, name, meta):
                self.path = Path(name)
                self.meta = meta

        class FakeCheckpointer:
            def __init__(self):
                self.calls = []

            def save_and_keep_only(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        checkpointer = FakeCheckpointer()
        original = checkpointer.save_and_keep_only
        with _best_only_retention(checkpointer):
            checkpointer.save_and_keep_only(
                meta={"si-snr": -18.3},
                min_keys=["si-snr"],
            )

        self.assertEqual(checkpointer.save_and_keep_only, original)
        _, kwargs = checkpointer.calls[0]
        self.assertFalse(kwargs["keep_recent"])
        self.assertEqual(kwargs["importance_keys"], [])
        predicate = kwargs["ckpt_predicate"]
        self.assertTrue(predicate(Checkpoint("CKPT+best", {"si-snr": -18.3})))
        self.assertFalse(predicate(Checkpoint("CKPT+last", {"si-snr": -18.1})))
        self.assertFalse(predicate(Checkpoint("CKPT+other", {"kind": "last"})))

    def test_speechbrain_best_retention_combines_existing_predicate(self):
        class Checkpoint:
            def __init__(self, name, eligible=True):
                self.path = Path(name)
                self.meta = {"eligible": eligible}

        class FakeCheckpointer:
            def save_and_keep_only(self, *args, **kwargs):
                self.kwargs = kwargs

        checkpointer = FakeCheckpointer()
        with _best_only_retention(checkpointer):
            checkpointer.save_and_keep_only(
                min_keys=["si-snr"],
                ckpt_predicate=lambda checkpoint: checkpoint.meta["eligible"],
            )

        predicate = checkpointer.kwargs["ckpt_predicate"]
        self.assertTrue(predicate(Checkpoint("CKPT+best")))
        self.assertFalse(predicate(Checkpoint("CKPT+ineligible", False)))
        self.assertFalse(predicate(Checkpoint("CKPT+last")))

    def test_clearvoice_pointer_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "custom"
            checkpoint.mkdir()
            (checkpoint / "last_best_checkpoint").write_text("model.pt\n")
            (checkpoint / "model.pt").write_bytes(b"weights")
            resolved = resolve_initial_checkpoint(root, "mossformer2", 2, checkpoint, root)
            self.assertEqual(resolved, checkpoint.resolve())
            (checkpoint / "model.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                resolve_initial_checkpoint(root, "mossformer2", 2, checkpoint, root)

    def test_common_export_overwrites_one_best_weight_file(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            bundle = ModelBundle(
                model=None,
                export_kind="convtasnet",
                export_metadata={"model_config": {"num_spks": 2}},
            )
            training = {
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "lr_factor": 0.5,
                "lr_patience": 2,
                "min_learning_rate": 1e-8,
            }
            export_model(bundle, run_dir, {"weight": torch.tensor([1.0])}, 1, 10, training)
            export_model(bundle, run_dir, {"weight": torch.tensor([2.0])}, 2, 20, training)
            weight_files = list((run_dir / "export").glob("*.tar"))
            self.assertEqual([path.name for path in weight_files], ["best.pt.tar"])
            payload = torch.load(weight_files[0], map_location="cpu", weights_only=False)
            self.assertEqual(payload["epoch"], 2)
            self.assertNotIn("optim_state_dict", payload)

    def test_speechbrain_best_is_moved_not_duplicated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            checkpoint = run_dir / "speechbrain" / "save" / "CKPT+best"
            checkpoint.mkdir(parents=True)
            for name in ("encoder.ckpt", "decoder.ckpt", "masknet.ckpt", "counter.ckpt"):
                (checkpoint / name).write_bytes(name.encode())
            initial = root / "initial"
            initial.mkdir()
            (initial / "hyperparams.yaml").write_text("sample_rate: 8000\n")
            output = export_speechbrain(
                {"run_dir": str(run_dir), "initial_checkpoint": str(initial)},
                checkpoint,
            )
            self.assertFalse(checkpoint.exists())
            self.assertEqual(
                sorted(path.name for path in output.glob("*.ckpt")),
                ["decoder.ckpt", "encoder.ckpt", "masknet.ckpt"],
            )


if __name__ == "__main__":
    unittest.main()
