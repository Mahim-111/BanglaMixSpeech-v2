from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import warnings

import numpy as np
import torch

try:
    from pesq import pesq as pesq_fn
    _HAS_PESQ = True
except Exception:
    pesq_fn = None
    _HAS_PESQ = False

try:
    from pystoi import stoi as stoi_fn
    _HAS_STOI = True
except Exception:
    stoi_fn = None
    _HAS_STOI = False

try:
    from mir_eval.separation import bss_eval_sources
    _HAS_MIR_EVAL = True
except Exception:
    bss_eval_sources = None
    _HAS_MIR_EVAL = False


METRIC_COLUMNS = [
    "SI-SDR",
    "SI-SDR_mixture",
    "SI-SDRi",
    "SDR",
    "SDR_mixture",
    "SDRi",
    "PESQ",
    "PESQ_mixture",
    "PESQi",
    "STOI",
    "STOI_mixture",
    "STOIi",
]


def metric_backend_status() -> Dict[str, object]:
    return {
        "pesq": _HAS_PESQ,
        "pystoi": _HAS_STOI,
        "mir_eval": _HAS_MIR_EVAL,
        "sdr_backend": "mir_eval" if _HAS_MIR_EVAL else "snr_fallback",
    }


def _to_1d_float_tensor(x) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.detach().cpu().float()
    if x.ndim == 1:
        return x.contiguous()
    if x.ndim == 2:
        return (x[0] if x.shape[0] <= x.shape[1] else x[:, 0]).contiguous()
    return x.flatten().contiguous()


def _align_1d(*signals: torch.Tensor) -> List[torch.Tensor]:
    length = min(int(x.numel()) for x in signals)
    return [x[:length].contiguous() for x in signals]


def si_sdr(estimate, reference, eps: float = 1e-8) -> float:
    est = _to_1d_float_tensor(estimate)
    ref = _to_1d_float_tensor(reference)
    est, ref = _align_1d(est, ref)

    est = est - est.mean()
    ref = ref - ref.mean()

    ref_energy = torch.sum(ref * ref) + eps
    projection = torch.sum(est * ref) * ref / ref_energy
    noise = est - projection

    value = 10.0 * torch.log10((torch.sum(projection * projection) + eps) / (torch.sum(noise * noise) + eps))
    return float(value)


def snr_sdr(estimate, reference, eps: float = 1e-8) -> float:
    est = _to_1d_float_tensor(estimate)
    ref = _to_1d_float_tensor(reference)
    est, ref = _align_1d(est, ref)

    est = est - est.mean()
    ref = ref - ref.mean()
    noise = est - ref
    value = 10.0 * torch.log10((torch.sum(ref * ref) + eps) / (torch.sum(noise * noise) + eps))
    return float(value)


def bss_sdr(estimate, reference) -> float:
    if not _HAS_MIR_EVAL:
        return snr_sdr(estimate, reference)

    est = _to_1d_float_tensor(estimate).numpy().astype(np.float64)
    ref = _to_1d_float_tensor(reference).numpy().astype(np.float64)
    est_t, ref_t = _align_1d(torch.from_numpy(est), torch.from_numpy(ref))
    ref_np = ref_t.numpy()[None, :]
    est_np = est_t.numpy()[None, :]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        sdr, _, _, _ = bss_eval_sources(ref_np, est_np, compute_permutation=False)
    return float(sdr[0])


def _as_numpy_audio(x, dtype=np.float64) -> np.ndarray:
    y = _to_1d_float_tensor(x).numpy().astype(dtype, copy=False)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(y)


def _align_np(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(int(a.shape[-1]), int(b.shape[-1]))
    return np.ascontiguousarray(a[:n]), np.ascontiguousarray(b[:n])


def _peak_normalize_for_intrusive_metric(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > eps:
        x = 0.99 * x / peak
    return np.ascontiguousarray(x.astype(np.float64, copy=False))


def _crop_to_active_reference(clean: np.ndarray, degraded: np.ndarray, sr: int, top_db: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
    """Crop long leading/trailing silence before STOI.

    BanglaMix mixtures can contain large non-overlap/silent regions, especially in low-overlap
    conditions. pystoi may return NaN when many frames are silent or when the degraded signal has
    zero-variance TF bands. Cropping by reference activity keeps the speech region and avoids letting
    padding silence dominate the STOI internals.
    """
    clean, degraded = _align_np(clean, degraded)
    if clean.size == 0:
        return clean, degraded

    frame_len = max(int(round(0.025 * sr)), 1)
    hop = max(int(round(0.010 * sr)), 1)
    if clean.size < frame_len:
        return clean, degraded

    starts = np.arange(0, clean.size - frame_len + 1, hop, dtype=np.int64)
    if starts.size == 0:
        return clean, degraded

    rms = np.empty(starts.size, dtype=np.float64)
    for i, st in enumerate(starts):
        frame = clean[st : st + frame_len]
        rms[i] = np.sqrt(np.mean(frame * frame) + 1e-20)

    max_rms = float(np.max(rms)) if rms.size else 0.0
    if max_rms <= 1e-10:
        return clean, degraded

    threshold = max(max_rms * (10.0 ** (-top_db / 20.0)), 1e-8)
    active = np.where(rms >= threshold)[0]
    if active.size == 0:
        return clean, degraded

    start = int(starts[int(active[0])])
    end = int(min(clean.size, starts[int(active[-1])] + frame_len))
    if end <= start:
        return clean, degraded

    return np.ascontiguousarray(clean[start:end]), np.ascontiguousarray(degraded[start:end])


def _signal_is_valid_for_stoi(x: np.ndarray, sr: int) -> bool:
    if x.size < max(int(0.4 * sr), 1):
        return False
    if not np.all(np.isfinite(x)):
        return False
    if float(np.sqrt(np.mean(x * x) + 1e-20)) <= 1e-10:
        return False
    if float(np.std(x)) <= 1e-10:
        return False
    return True


def safe_stoi(clean, degraded, sr: int) -> float:
    """Robust STOI wrapper.

    pystoi sometimes returns NaN instead of raising an exception when silent regions or zero-variance
    bands are present. This wrapper retries after active-speech cropping and peak normalization. If
    the reference is valid but the estimate is effectively silent/invalid, it returns 0.0 instead of
    NaN because intelligibility is practically zero. If pystoi is not installed or the reference is
    invalid, it returns NaN.
    """
    if not _HAS_STOI:
        return float("nan")

    clean_np = _as_numpy_audio(clean, dtype=np.float64)
    degraded_np = _as_numpy_audio(degraded, dtype=np.float64)
    clean_np, degraded_np = _align_np(clean_np, degraded_np)

    if not _signal_is_valid_for_stoi(clean_np, sr):
        return float("nan")

    attempts = []

    c0 = _peak_normalize_for_intrusive_metric(clean_np)
    d0 = _peak_normalize_for_intrusive_metric(degraded_np)
    attempts.append((c0, d0))

    c1, d1 = _crop_to_active_reference(clean_np, degraded_np, sr=sr, top_db=60.0)
    c1 = _peak_normalize_for_intrusive_metric(c1)
    d1 = _peak_normalize_for_intrusive_metric(d1)
    attempts.append((c1, d1))

    c2, d2 = _crop_to_active_reference(clean_np, degraded_np, sr=sr, top_db=40.0)
    c2 = _peak_normalize_for_intrusive_metric(c2)
    d2 = _peak_normalize_for_intrusive_metric(d2)
    attempts.append((c2, d2))

    degraded_valid = any(_signal_is_valid_for_stoi(d, sr) for _, d in attempts)

    for clean_try, degraded_try in attempts:
        if not _signal_is_valid_for_stoi(clean_try, sr):
            continue
        if not _signal_is_valid_for_stoi(degraded_try, sr):
            continue
        try:
            value = float(stoi_fn(clean_try, degraded_try, sr, extended=False))
            if np.isfinite(value):
                return value
        except Exception:
            continue

    if not degraded_valid:
        return 0.0

    # Final practical fallback: tiny deterministic dither avoids zero-variance sub-band divisions
    # inside pystoi. The level is far below the speech amplitude after peak normalization.
    try:
        rng = np.random.default_rng(0)
        c, d = attempts[-1]
        if _signal_is_valid_for_stoi(c, sr) and d.size == c.size:
            d = np.ascontiguousarray(d + rng.normal(0.0, 1e-10, size=d.shape))
            value = float(stoi_fn(c, d, sr, extended=False))
            if np.isfinite(value):
                return value
    except Exception:
        pass

    return float("nan")


class SeparationMetrics:
    """Compute speech-separation metrics for one estimate/reference pair.

    mode="fast" computes SI-SDR/SDR only.
    mode="full" also computes PESQ/STOI and their mixture baselines.
    """

    def __init__(self, mode: str = "full"):
        mode = mode.lower().strip()
        if mode not in {"fast", "full"}:
            raise ValueError("mode must be 'fast' or 'full'")
        self.mode = mode
        self.sdr_backend = "mir_eval" if _HAS_MIR_EVAL else "snr_fallback"

        if self.mode == "full":
            missing = []
            if not _HAS_PESQ:
                missing.append("pesq")
            if not _HAS_STOI:
                missing.append("pystoi")
            if missing:
                warnings.warn(
                    "Full metric mode requested but missing package(s): "
                    + ", ".join(missing)
                    + ". Corresponding CSV columns will be NaN.",
                    RuntimeWarning,
                )

    def compute_pair(self, estimate, target, mixture, sr: int = 16000) -> Dict[str, float]:
        estimate = _to_1d_float_tensor(estimate)
        target = _to_1d_float_tensor(target)
        mixture = _to_1d_float_tensor(mixture)
        estimate, target, mixture = _align_1d(estimate, target, mixture)

        sisdr_est = si_sdr(estimate, target)
        sisdr_mix = si_sdr(mixture, target)

        try:
            sdr_est = bss_sdr(estimate, target)
            sdr_mix = bss_sdr(mixture, target)
        except Exception:
            sdr_est = float("nan")
            sdr_mix = float("nan")

        out = {
            "SI-SDR": sisdr_est,
            "SI-SDR_mixture": sisdr_mix,
            "SI-SDRi": sisdr_est - sisdr_mix,
            "SDR": sdr_est,
            "SDR_mixture": sdr_mix,
            "SDRi": sdr_est - sdr_mix if np.isfinite(sdr_est) and np.isfinite(sdr_mix) else float("nan"),
            "PESQ": float("nan"),
            "PESQ_mixture": float("nan"),
            "PESQi": float("nan"),
            "STOI": float("nan"),
            "STOI_mixture": float("nan"),
            "STOIi": float("nan"),
        }

        if self.mode != "full":
            return out

        target_np = _as_numpy_audio(target, dtype=np.float64)
        estimate_np = _as_numpy_audio(estimate, dtype=np.float64)
        mixture_np = _as_numpy_audio(mixture, dtype=np.float64)

        if _HAS_PESQ and sr in (8000, 16000):
            mode = "wb" if sr == 16000 else "nb"
            try:
                # PESQ expects normal audio-like scale. Normalize only for PESQ numerical stability;
                # the intrusive score is still computed between target and estimate.
                pesq_target = _peak_normalize_for_intrusive_metric(target_np).astype(np.float32)
                pesq_estimate = _peak_normalize_for_intrusive_metric(estimate_np).astype(np.float32)
                pesq_mixture = _peak_normalize_for_intrusive_metric(mixture_np).astype(np.float32)
                pesq_est = float(pesq_fn(sr, pesq_target, pesq_estimate, mode))
                pesq_mix = float(pesq_fn(sr, pesq_target, pesq_mixture, mode))
                out["PESQ"] = pesq_est
                out["PESQ_mixture"] = pesq_mix
                out["PESQi"] = pesq_est - pesq_mix
            except Exception:
                pass

        if _HAS_STOI:
            stoi_est = safe_stoi(target_np, estimate_np, sr)
            stoi_mix = safe_stoi(target_np, mixture_np, sr)
            out["STOI"] = stoi_est
            out["STOI_mixture"] = stoi_mix
            out["STOIi"] = stoi_est - stoi_mix if np.isfinite(stoi_est) and np.isfinite(stoi_mix) else float("nan")

        return out

    @staticmethod
    def average_dicts(dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
        dicts = list(dicts)
        avg: Dict[str, float] = {}
        for key in METRIC_COLUMNS:
            values = np.asarray([d.get(key, np.nan) for d in dicts], dtype=np.float64)
            finite = values[np.isfinite(values)]
            avg[key] = float(finite.mean()) if finite.size else float("nan")
        return avg
