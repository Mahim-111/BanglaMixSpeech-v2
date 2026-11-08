from __future__ import annotations

from pathlib import Path
import contextlib
import io
import sys

from checkpointing import checkpoint_fingerprint
from utils import PROJECT_ROOT, normalize_sources_tensor, write_sources

BUNDLED_SPEECHBRAIN = PROJECT_ROOT / "repos" / "speechbrain"
if BUNDLED_SPEECHBRAIN.is_dir() and str(BUNDLED_SPEECHBRAIN) not in sys.path:
    sys.path.insert(0, str(BUNDLED_SPEECHBRAIN))


def _speechbrain_cache_name(
    checkpoint_name: str,
    source_dir: Path,
    speakers: int,
) -> str:
    """Build a cache name that changes whenever checkpoint content changes."""
    content_digest = checkpoint_fingerprint("sepformer", speakers, source_dir)
    return f"{checkpoint_name}__{content_digest[:16]}"


class SpeechBrainAdapter:
    """SpeechBrain SepFormer adapter.

    Variants:
    - wsj0_8k   : original 8 kHz WSJ0 2/3-speaker baselines.
    - whamr16k  : 16 kHz WHAMR 2-speaker SepFormer.

    The 16 kHz variant is intentionally exposed as model name `sepformer16k`
    through models.py so it does not overwrite old `sepformer` results.
    """

    def __init__(
        self,
        speakers: int = 2,
        device: str = "cuda:0",
        verbose: bool = False,
        variant: str = "wsj0_8k",
        checkpoint_dir: str | Path | None = None,
    ):
        # Keep optional SpeechBrain dependencies lazy so checkpoint/cache logic
        # remains testable without importing the full inference environment.
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.utils.fetching import FetchConfig, LocalStrategy

        self.speakers = int(speakers)
        self.device = device
        self.verbose = bool(verbose)
        self.variant = variant.lower().strip()

        if self.variant == "whamr16k":
            if self.speakers != 2:
                raise ValueError(
                    "sepformer16k uses speechbrain/sepformer-whamr16k, which is a 2-speaker model only."
                )
            self.name = "sepformer16k"
            self.sample_rate = 16000
            source = "speechbrain/sepformer-whamr16k"

        elif self.variant == "wsj0_8k":
            self.name = "sepformer"
            self.sample_rate = 8000
            if self.speakers == 2:
                source = "speechbrain/sepformer-wsj02mix"
            elif self.speakers == 3:
                source = "speechbrain/sepformer-wsj03mix"
            else:
                raise ValueError("SpeechBrain SepFormer supports only 2 or 3 speakers.")

        else:
            raise ValueError(f"Unknown SpeechBrain SepFormer variant: {variant}")

        self.source = source
        checkpoint_name = source.split("/")[-1]
        source_dir = (
            Path(checkpoint_dir).expanduser().resolve()
            if checkpoint_dir is not None
            else PROJECT_ROOT / "checkpoints" / checkpoint_name
        )
        required = [
            source_dir / "hyperparams.yaml",
            source_dir / "encoder.ckpt",
            source_dir / "decoder.ckpt",
            source_dir / "masknet.ckpt",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Bundled SepFormer checkpoint is incomplete:\n  "
                + "\n  ".join(missing)
                + "\nRun: python scripts/verify_bundle.py --full"
            )

        self.source = str(source_dir)
        # SpeechBrain does not overwrite files already present in ``savedir``.
        # A path-only cache key therefore loads stale weights when a fine-tuned
        # checkpoint folder is replaced in place. Address the cache by the same
        # checkpoint-content fingerprint recorded in benchmark result rows.
        cache_name = _speechbrain_cache_name(
            checkpoint_name,
            source_dir,
            self.speakers,
        )
        self.savedir = PROJECT_ROOT / "cache" / "speechbrain" / cache_name
        self.savedir.mkdir(parents=True, exist_ok=True)

        self.model = SepformerSeparation.from_hparams(
            source=self.source,
            savedir=str(self.savedir),
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY,
            fetch_config=FetchConfig(allow_network=False),
        )

    def separate(self, mixture_path, output_dir=None):
        path = str(Path(mixture_path).expanduser().resolve())
        if self.verbose:
            estimates = self.model.separate_file(path=path)
        else:
            # Keep benchmark CLI clean unless --verbose is requested.
            with contextlib.redirect_stdout(io.StringIO()):
                estimates = self.model.separate_file(path=path)
        return normalize_sources_tensor(estimates, expected_speakers=self.speakers)

    def separate_and_save(self, mixture_path, output_dir):
        estimates = self.separate(mixture_path)
        return write_sources(output_dir, estimates, self.sample_rate)
