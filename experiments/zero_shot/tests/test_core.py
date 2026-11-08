from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from dataset import BanglaMixDataset
from checkpointing import (
    CheckpointNotFoundError,
    InvalidCheckpointError,
    checkpoint_fingerprint,
    resolve_checkpoint_from_root,
    result_model_name,
)
from evaluate import Evaluator
from models import supported_speakers
from adapters.speechbrain_adapter import _speechbrain_cache_name


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)


class DummySeparator:
    name = "dummy"
    sample_rate = 16000

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def separate(self, mixture_path, output_dir=None):
        self.calls += 1
        if self.fail:
            raise AssertionError("resume should not call the model")
        audio, _ = sf.read(mixture_path, dtype="float32")
        return torch.from_numpy(np.stack([audio, audio]))


class CoreTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        rng = np.random.default_rng(7)
        s1 = rng.normal(0, 0.02, 3200).astype(np.float32)
        s2 = rng.normal(0, 0.02, 3200).astype(np.float32)
        sample = (
            root
            / "Mix_Speech"
            / "Mix_For_Two_Speaker"
            / "test"
            / "overlap_100%"
            / "item001"
        )
        write_wav(sample / "s1.wav", s1)
        write_wav(sample / "s2.wav", s2)
        write_wav(sample / "mix.wav", s1 + s2)

        sample3 = (
            root
            / "Mix_Speech"
            / "Mix_For_Three_Speaker"
            / "max_overlap_jitter12pct"
            / "test"
            / "item003"
        )
        write_wav(sample3 / "s1.wav", s1)
        write_wav(sample3 / "s2.wav", s2)
        write_wav(sample3 / "s3.wav", 0.5 * s1)
        write_wav(sample3 / "mix.wav", 1.5 * s1 + s2)
        return root

    def test_dataset_layout_and_model_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = BanglaMixDataset(self.make_dataset(Path(tmp)))
            self.assertEqual(dataset.list_variations(2, "test"), ["overlap_100%"])
            self.assertEqual(
                dataset.list_variations(3, "test"), ["max_overlap_jitter12pct"]
            )
            self.assertEqual(len(dataset.load(2, "test", "overlap_100%")), 1)
        self.assertEqual(supported_speakers("mossformer2"), (2, 3))
        self.assertEqual(supported_speakers("convtasnet"), (2,))

    def test_evaluator_resumes_completed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_dataset(Path(tmp) / "data")
            dataset = BanglaMixDataset(root)
            samples = dataset.load(2, "test", "overlap_100%")
            csv_path = Path(tmp) / "results.csv"

            first_model = DummySeparator()
            Evaluator(metric_mode="fast", progress=False, resume=True).evaluate_dataset(
                first_model,
                samples,
                save_csv=csv_path,
                model_name="dummy",
                split="test",
                variation="overlap_100%",
            )
            self.assertEqual(first_model.calls, 1)
            self.assertEqual(len(pd.read_csv(csv_path)), 1)

            resumed_model = DummySeparator(fail=True)
            Evaluator(metric_mode="fast", progress=False, resume=True).evaluate_dataset(
                resumed_model,
                samples,
                save_csv=csv_path,
                model_name="dummy",
                split="test",
                variation="overlap_100%",
            )
            self.assertEqual(resumed_model.calls, 0)
            self.assertEqual(len(pd.read_csv(csv_path)), 1)

            changed_protocol_model = DummySeparator()
            Evaluator(
                metric_mode="fast",
                progress=False,
                resume=True,
                evaluation_sr="model",
            ).evaluate_dataset(
                changed_protocol_model,
                samples,
                save_csv=csv_path,
                model_name="dummy",
                split="test",
                variation="overlap_100%",
            )
            self.assertEqual(changed_protocol_model.calls, 1)
            self.assertEqual(
                pd.read_csv(csv_path).loc[0, "evaluation_sr_policy"], "model"
            )

    def test_custom_checkpoint_layout_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment_alpha"
            checkpoint = root / "dprnn" / "2spk" / "pytorch_model.bin"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"first fine-tuned state")

            resolved = resolve_checkpoint_from_root(root, "dprnn", 2)
            self.assertEqual(resolved, checkpoint.resolve())
            first_fingerprint = checkpoint_fingerprint("dprnn", 2, resolved)
            self.assertEqual(
                result_model_name("dprnn", resolved, None),
                "dprnn__experiment_alpha",
            )

            checkpoint.write_bytes(b"replacement fine-tuned state")
            second_fingerprint = checkpoint_fingerprint("dprnn", 2, resolved)
            self.assertNotEqual(first_fingerprint, second_fingerprint)

            clearvoice = root / "mossformer2" / "2spk"
            clearvoice.mkdir(parents=True)
            model_file = clearvoice / "model.ckpt-42.pt"
            model_file.write_bytes(b"fine-tuned ClearVoice weights")
            (clearvoice / "last_best_checkpoint").write_text(
                f"{model_file.name}\n", encoding="utf-8"
            )
            self.assertEqual(
                resolve_checkpoint_from_root(root, "mossformer2", 2),
                clearvoice.resolve(),
            )
            clearvoice_fingerprint = checkpoint_fingerprint(
                "mossformer2", 2, clearvoice
            )
            model_file.write_bytes(b"replacement ClearVoice weights")
            self.assertNotEqual(
                clearvoice_fingerprint,
                checkpoint_fingerprint("mossformer2", 2, clearvoice),
            )

            with self.assertRaises(CheckpointNotFoundError):
                resolve_checkpoint_from_root(root, "sepformer", 3)

            malformed = root / "sepformer" / "2spk"
            malformed.mkdir(parents=True)
            (malformed / "encoder.ckpt").write_bytes(b"incomplete")
            with self.assertRaises(InvalidCheckpointError):
                resolve_checkpoint_from_root(root, "sepformer", 2)

    def test_sepformer_cache_key_changes_with_checkpoint_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "sepformer" / "2spk"
            checkpoint.mkdir(parents=True)
            for name in (
                "hyperparams.yaml",
                "encoder.ckpt",
                "decoder.ckpt",
                "masknet.ckpt",
            ):
                (checkpoint / name).write_bytes(f"first {name}".encode())

            first = _speechbrain_cache_name(
                "sepformer-wsj02mix",
                checkpoint,
                speakers=2,
            )
            (checkpoint / "masknet.ckpt").write_bytes(b"replacement weights")
            second = _speechbrain_cache_name(
                "sepformer-wsj02mix",
                checkpoint,
                speakers=2,
            )

            self.assertNotEqual(first, second)
            self.assertTrue(first.startswith("sepformer-wsj02mix__"))


if __name__ == "__main__":
    unittest.main()
