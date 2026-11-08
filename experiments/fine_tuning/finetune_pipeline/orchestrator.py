from __future__ import annotations

import codecs
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import ResolvedJob, expand_jobs, resolve_output_root, resolve_project_root
from .manifests import prepare_manifests
from .registry import get_model_spec, resolve_initial_checkpoint
from .util import atomic_write_json, process_is_alive, safe_slug, stable_hash


SEPFORMER_TRAINER_IMPLEMENTATION = "model_rate_resampling_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_files(initial: Path, job: ResolvedJob) -> list[Path]:
    spec = get_model_spec(job.model, job.speakers)
    files = [initial / name for name in spec.required_checkpoint_files]
    if spec.backend == "torch_mossformer2_clearvoice":
        pointer = initial / "last_best_checkpoint"
        target = pointer.read_text(encoding="utf-8").splitlines()[0].strip()
        files.append(initial / target)
    return sorted(set(files), key=lambda path: str(path))


def checkpoint_fingerprint(initial: Path, job: ResolvedJob) -> str:
    digest = hashlib.sha256()
    for path in _checkpoint_files(initial, job):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def resolve_dataset_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    dataset = dict(config["dataset"])
    root = Path(dataset["root"]).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    dataset["root"] = str(root.resolve())
    return dataset


def prepare_for_jobs(
    config: dict[str, Any],
    config_path: Path,
    jobs: Iterable[ResolvedJob],
    force: bool = False,
) -> dict[int, dict[str, Any]]:
    output_root = resolve_output_root(config, config_path)
    dataset = resolve_dataset_config(config, config_path)
    manifest_root = output_root / safe_slug(config["experiment"]["name"]) / "_manifests"
    return {
        speakers: prepare_manifests(dataset, speakers, manifest_root, force=force)
        for speakers in sorted({job.speakers for job in jobs})
    }


def _resolved_run(
    config: dict[str, Any],
    config_path: Path,
    job: ResolvedJob,
    manifest: dict[str, Any],
    device: str,
    fingerprint_cache: dict[tuple[str, int, str], str],
    cache_lock: threading.Lock,
) -> tuple[Path, dict[str, Any]]:
    project_root = resolve_project_root(config, config_path)
    output_root = resolve_output_root(config, config_path)
    initial = resolve_initial_checkpoint(
        project_root,
        job.model,
        job.speakers,
        job.initial_checkpoint_value,
        config_path.parent,
    )
    cache_key = (job.model, job.speakers, str(initial))
    with cache_lock:
        initial_fingerprint = fingerprint_cache.get(cache_key)
    if initial_fingerprint is None:
        computed = checkpoint_fingerprint(initial, job)
        with cache_lock:
            initial_fingerprint = fingerprint_cache.setdefault(cache_key, computed)
    execution_identity = {
        "job": job.identity(),
        "manifest_fingerprint": manifest["fingerprint"],
        "initial_checkpoint_fingerprint": initial_fingerprint,
    }
    trainer_implementation = None
    if job.backend == "speechbrain_sepformer":
        # Keep corrected SepFormer runs separate from older completed runs that
        # loaded 16 kHz BanglaMix WAVs without converting them to the model's
        # native 8 kHz rate.
        trainer_implementation = SEPFORMER_TRAINER_IMPLEMENTATION
        execution_identity["trainer_implementation"] = trainer_implementation
    execution_hash = stable_hash(execution_identity)
    label_prefix = job.run_label.rsplit("__", 1)[0]
    run_label = f"{label_prefix}__{execution_hash}"
    run_dir = (
        output_root
        / safe_slug(config["experiment"]["name"])
        / job.model
        / f"{job.speakers}spk"
        / run_label
    )
    outputs = manifest["outputs"]
    resolved = {
        "version": 1,
        "experiment": config["experiment"]["name"],
        "model": job.model,
        "speakers": job.speakers,
        "backend": job.backend,
        "sample_rate": job.sample_rate,
        "seed": job.seed,
        "repeat": job.repeat,
        "training": job.training,
        "sweep_values": job.sweep_values,
        "device": device,
        "run_hash": execution_hash,
        "run_label": run_label,
        "run_dir": str(run_dir.resolve()),
        "project_root": str(project_root),
        "pipeline_root": str(Path(__file__).resolve().parents[1]),
        "dataset_root": resolve_dataset_config(config, config_path)["root"],
        "manifest_fingerprint": manifest["fingerprint"],
        "manifest_summary": str(Path(manifest["folder"]) / "manifest_summary.json"),
        "manifests": {
            "train": outputs["train"]["jsonl"],
            "valid": outputs["valid"]["jsonl"],
            "test": outputs["test"]["jsonl"],
            "train_speechbrain": outputs["train"]["speechbrain_csv"],
            "valid_speechbrain": outputs["valid"]["speechbrain_csv"],
            "test_speechbrain": outputs["test"]["speechbrain_csv"],
        },
        "initial_checkpoint": str(initial),
        "initial_checkpoint_fingerprint": initial_fingerprint,
        "trainer_implementation": trainer_implementation,
        "checkpoint_retention": "last_and_last_best_only",
        "source_config": str(config_path),
    }
    return run_dir, resolved


def build_runs(
    config: dict[str, Any],
    config_path: Path,
    jobs: list[ResolvedJob],
    manifests: dict[int, dict[str, Any]],
    devices: list[str],
) -> list[tuple[Path, dict[str, Any]]]:
    cache: dict[tuple[str, int, str], str] = {}
    cache_lock = threading.Lock()
    return [
        _resolved_run(
            config, config_path, job, manifests[job.speakers], devices[index % len(devices)], cache, cache_lock
        )
        for index, job in enumerate(jobs)
    ]


def _export_complete(run_dir: Path) -> bool:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    export_dir = run_dir / "export"
    return status.get("state") == "completed" and export_dir.is_dir() and any(export_dir.iterdir())


def _claim_lock(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / ".run.lock"
    payload = {"pid": os.getpid(), "created_at": utc_now()}
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            existing = json.loads(lock.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        pid = int(existing.get("pid", -1))
        if process_is_alive(pid):
            raise RuntimeError(f"Run is already locked by live process {pid}: {run_dir}")
        lock.unlink()
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return lock


def _run_and_tee(command: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    pipeline_root = str(Path(__file__).resolve().parents[1])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = pipeline_root + (os.pathsep + existing if existing else "")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert process.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        with process.stdout:
            while True:
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                sys.stdout.write(text)
                sys.stdout.flush()
                log.write(text)
                log.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            sys.stdout.write(tail)
            sys.stdout.flush()
            log.write(tail)
            log.flush()
    return process.wait()


def run_one(run_dir: Path, resolved: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    if _export_complete(run_dir):
        return {"run_dir": str(run_dir), "state": "skipped", "reason": "already completed"}
    if dry_run:
        return {"run_dir": str(run_dir), "state": "planned", "device": resolved["device"]}
    lock = _claim_lock(run_dir)
    status_path = run_dir / "status.json"
    config_path = run_dir / "resolved_run.json"
    started = time.time()
    status = {
        "state": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "model": resolved["model"],
        "speakers": resolved["speakers"],
        "run_hash": resolved["run_hash"],
        "device": resolved["device"],
        "pid": os.getpid(),
    }
    try:
        atomic_write_json(config_path, resolved)
        atomic_write_json(status_path, status)
        module = (
            "finetune_pipeline.speechbrain_trainer"
            if resolved["backend"] == "speechbrain_sepformer"
            else "finetune_pipeline.torch_trainer"
        )
        command = [sys.executable, "-m", module, "--run-config", str(config_path)]
        code = _run_and_tee(command, Path(resolved["pipeline_root"]), run_dir / "train.log")
        if code != 0:
            raise RuntimeError(f"Trainer exited with status {code}; see {run_dir / 'train.log'}")
        status.update(
            state="completed",
            completed_at=utc_now(),
            updated_at=utc_now(),
            elapsed_seconds=time.time() - started,
            export_dir=str((run_dir / "export").resolve()),
        )
        atomic_write_json(status_path, status)
        return {"run_dir": str(run_dir), "state": "completed"}
    except KeyboardInterrupt:
        status.update(state="interrupted", updated_at=utc_now(), elapsed_seconds=time.time() - started)
        atomic_write_json(status_path, status)
        raise
    except Exception as exc:
        status.update(
            state="failed",
            updated_at=utc_now(),
            elapsed_seconds=time.time() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(status_path, status)
        return {"run_dir": str(run_dir), "state": "failed", "error": str(exc)}
    finally:
        if lock.exists():
            lock.unlink()


def execute_runs(
    runs: list[tuple[Path, dict[str, Any]]],
    parallel: int = 1,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    if parallel <= 0:
        raise ValueError("parallel must be positive")
    if parallel == 1:
        results = []
        for run_dir, resolved in runs:
            result = run_one(run_dir, resolved, dry_run=dry_run)
            results.append(result)
            if fail_fast and result["state"] == "failed":
                break
        return results
    results = []
    device_locks = {
        resolved["device"]: threading.Semaphore(1)
        for _, resolved in runs
    }

    def run_with_device_lock(run_dir: Path, resolved: dict[str, Any]):
        with device_locks[resolved["device"]]:
            return run_one(run_dir, resolved, dry_run=dry_run)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(run_with_device_lock, run_dir, resolved): run_dir
            for run_dir, resolved in runs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if fail_fast and result["state"] == "failed":
                for pending in futures:
                    pending.cancel()
                break
    return results


def collect_status(output_root: Path, experiment: str | None = None) -> list[dict[str, Any]]:
    base = output_root / safe_slug(experiment) if experiment else output_root
    rows = []
    if not base.exists():
        return rows
    for path in sorted(base.glob("**/status.json")):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            status = {"state": "invalid", "error": str(exc)}
        status["run_dir"] = str(path.parent)
        rows.append(status)
    return rows
