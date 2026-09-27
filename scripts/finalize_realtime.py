#!/usr/bin/env python3
"""Verify, document, archive, and optionally publish the realtime_stage1 suite."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SUITE = "realtime_stage1"
NEW_PREFIXES = (
    f"results/{SUITE}/",
    f"checkpoints/{SUITE}/",
    f"assets/{SUITE}/",
)
PUBLISH_FILES = {"README.md", "results/artifact_manifest.json"}
README_START = "<!-- realtime-stage1:start -->"
README_END = "<!-- realtime-stage1:end -->"
EXPECTED_ARMS = 5
EXPECTED_EPOCHS = 100
EXPECTED_LEGACY_ARTIFACTS = 290


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        stream.write(value)
        pending = Path(stream.name)
    pending.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def verify_training_batches(
    metrics: dict[str, Any], arm: dict[str, Any], training: dict[str, Any], arm_id: str,
) -> tuple[int, int]:
    expected_physical = arm.get("batch", training.get("batch"))
    expected_effective = training.get("nbs")
    if not isinstance(expected_physical, int) or expected_physical < 1:
        raise ValueError(f"{arm_id}: protocol has invalid physical batch {expected_physical!r}")
    if not isinstance(expected_effective, int) or expected_effective < expected_physical:
        raise ValueError(f"{arm_id}: protocol has invalid effective batch {expected_effective!r}")
    physical = metrics.get("batch")
    effective = metrics.get("effective_batch", metrics.get("effective_batch_size"))
    if physical != expected_physical:
        raise ValueError(
            f"{arm_id}: metrics physical batch {physical!r} differs from protocol {expected_physical}"
        )
    if effective != expected_effective:
        raise ValueError(
            f"{arm_id}: metrics effective batch {effective!r} differs from protocol {expected_effective}"
        )
    return physical, effective


def verify_epochs(path: Path, selected_epoch: Any, expected: int) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected:
        raise ValueError(f"{path} has {len(rows)} rows, expected {expected}")
    epochs = [int(finite_number(row.get("epoch"), f"{path}: epoch")) for row in rows]
    if epochs != list(range(1, expected + 1)):
        raise ValueError(f"{path} does not contain continuous epochs 1..{expected}")
    if not isinstance(selected_epoch, int) or not 1 <= selected_epoch <= expected:
        raise ValueError(f"invalid selected_epoch {selected_epoch!r} in {path.parent}")
    key = "metrics/mAP50-95(B)"
    aps = [finite_number(row.get(key), f"{path}: {key}") for row in rows]
    best = max(aps)
    selected = aps[selected_epoch - 1]
    # Ultralytics serializes CSV metrics at limited decimal precision.
    if not math.isclose(selected, best, rel_tol=1e-9, abs_tol=5e-7):
        raise ValueError(
            f"selected_epoch {selected_epoch} has AP50-95 {selected}, below CSV best {best}"
        )
    return {"selected_csv_ap50_95": selected, "best_csv_ap50_95": best}


def verify_latency(
    path: Path, *, arm: dict[str, Any], checkpoint_hash: str,
    metrics_hash: str, protocol_hash: str,
) -> dict[str, float]:
    value = read_object(path)
    expected = {
        "schema_version": 1,
        "status": "completed",
        "arm_id": arm["id"],
        "architecture": arm["architecture"],
        "imgsz": arm["imgsz"],
        "seed": arm["seed"],
        "checkpoint_sha256": checkpoint_hash,
        "precision": "FP16",
        "fused": True,
        "batch": 1,
        "warmup_iterations": 30,
        "timed_iterations": 100,
        "settings": {"conf": 0.25, "iou": 0.7, "max_det": 500, "rect": False},
        "metrics_json_sha256": metrics_hash,
        "protocol_json_sha256": protocol_hash,
    }
    mismatches = [key for key, wanted in expected.items() if value.get(key) != wanted]
    if mismatches:
        raise ValueError(f"{path} provenance/settings mismatch: {mismatches}")
    summaries: dict[str, dict[str, float]] = {}
    for name in ("pure_model_cuda_event", "synchronized_wall_clock"):
        raw = value.get(name)
        if not isinstance(raw, dict) or raw.get("samples") != 100:
            raise ValueError(f"{path}: invalid {name} sample count")
        summary = {
            key: finite_number(raw.get(key), f"{path}: {name}.{key}")
            for key in ("mean_ms", "p50_ms", "p95_ms", "fps_from_mean")
        }
        if any(number <= 0 for number in summary.values()):
            raise ValueError(f"{path}: non-positive {name} value")
        if summary["p95_ms"] < summary["p50_ms"]:
            raise ValueError(f"{path}: {name} p95 is below p50")
        if not math.isclose(summary["fps_from_mean"], 1000.0 / summary["mean_ms"], rel_tol=2e-4):
            raise ValueError(f"{path}: {name} FPS does not match mean latency")
        summaries[name] = summary
    return summaries["synchronized_wall_clock"]


def verify_suite(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite_dir = root / "results" / SUITE
    protocol_path = suite_dir / "protocol.json"
    protocol = read_object(protocol_path)
    arms = protocol.get("arms")
    if not isinstance(arms, list) or len(arms) != EXPECTED_ARMS:
        raise ValueError(f"protocol must contain exactly {EXPECTED_ARMS} arms")
    if protocol.get("training", {}).get("epochs") != EXPECTED_EPOCHS:
        raise ValueError(f"protocol epoch budget must be {EXPECTED_EPOCHS}")
    training = protocol["training"]
    arm_ids = [arm.get("id") for arm in arms if isinstance(arm, dict)]
    if len(arm_ids) != EXPECTED_ARMS or len(set(arm_ids)) != EXPECTED_ARMS:
        raise ValueError("protocol arms must be objects with unique ids")
    protocol_canonical_hash = canonical_sha(protocol)
    protocol_file_hash = sha256(protocol_path)
    verified: list[dict[str, Any]] = []
    for arm in arms:
        arm_id = arm["id"]
        if not isinstance(arm_id, str) or PurePosixPath(arm_id).name != arm_id:
            raise ValueError(f"unsafe arm id {arm_id!r}")
        run_dir = suite_dir / arm_id
        metrics_path = run_dir / "metrics.json"
        metrics = read_object(metrics_path)
        expected_identity = {"protocol_sha256": protocol_canonical_hash, "arm": arm}
        if metrics.get("status") != "completed" or metrics.get("experiment_identity") != expected_identity:
            raise ValueError(f"{arm_id}: incomplete or incompatible metrics identity")
        for key in ("architecture", "imgsz", "seed"):
            if metrics.get(key) != arm.get(key):
                raise ValueError(f"{arm_id}: metrics {key} differs from protocol")
        if metrics.get("epochs") != EXPECTED_EPOCHS:
            raise ValueError(f"{arm_id}: metrics epoch budget is not {EXPECTED_EPOCHS}")
        checkpoint = root / "checkpoints" / SUITE / f"{arm_id}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hash = sha256(checkpoint)
        if metrics.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"{arm_id}: checkpoint SHA-256 mismatch")
        epoch_check = verify_epochs(run_dir / "epochs.csv", metrics.get("selected_epoch"), EXPECTED_EPOCHS)
        metric_values = metrics.get("metrics")
        if not isinstance(metric_values, dict):
            raise ValueError(f"{arm_id}: missing metrics object")
        map50_95 = finite_number(metric_values.get("metrics/mAP50-95(B)"), f"{arm_id}: AP50-95")
        map50 = finite_number(metric_values.get("metrics/mAP50(B)"), f"{arm_id}: AP50")
        finite_number(metric_values.get("metrics/precision(B)"), f"{arm_id}: precision")
        finite_number(metric_values.get("metrics/recall(B)"), f"{arm_id}: recall")
        physical_batch, effective_batch = verify_training_batches(metrics, arm, training, arm_id)
        selection = metrics.get("selection_metrics")
        if not isinstance(selection, dict):
            raise ValueError(f"{arm_id}: missing selection_metrics")
        selection_ap = finite_number(selection.get("csv_ap50_95"), f"{arm_id}: selected CSV AP50-95")
        final_delta = finite_number(
            selection.get("final_minus_selection_ap50_95"), f"{arm_id}: final-selection AP50-95 delta"
        )
        if not math.isclose(
            selection_ap, epoch_check["selected_csv_ap50_95"], rel_tol=1e-9, abs_tol=5e-7
        ):
            raise ValueError(f"{arm_id}: selection_metrics does not match the selected CSV epoch")
        # Final evaluation reloads and fuses the selected checkpoint. Its AP may
        # differ slightly from the unfused EMA validation used for epoch selection.
        if not math.isclose(final_delta, map50_95 - selection_ap, rel_tol=1e-9, abs_tol=5e-9):
            raise ValueError(f"{arm_id}: final-selection AP50-95 delta is inconsistent")
        latency = verify_latency(
            run_dir / "latency.json", arm=arm, checkpoint_hash=checkpoint_hash,
            metrics_hash=sha256(metrics_path), protocol_hash=protocol_file_hash,
        )
        verified.append({
            "id": arm_id, "architecture": arm["architecture"], "imgsz": arm["imgsz"],
            "map50_95": map50_95, "map50": map50,
            "physical_batch": physical_batch, "effective_batch": effective_batch,
            "latency_p50_ms": latency["p50_ms"], "latency_p95_ms": latency["p95_ms"],
            "latency_fps": latency["fps_from_mean"],
        })
    queue = read_object(suite_dir / "queue_state.json")
    if queue.get("status") != "completed" or queue.get("suite") != SUITE:
        raise ValueError("queue_state.json is not a stable completed suite state")
    return protocol, verified


def verify_report(root: Path, verified: list[dict[str, Any]]) -> None:
    suite_dir = root / "results" / SUITE
    summary = read_object(suite_dir / "summary.json")
    counts = summary.get("status_counts")
    if (summary.get("suite") != SUITE or summary.get("expected_arms") != EXPECTED_ARMS
            or summary.get("completed_arms") != EXPECTED_ARMS
            or counts != {"completed": EXPECTED_ARMS, "running": 0, "pending": 0, "failed": 0, "invalid": 0}):
        raise ValueError("summary.json does not describe a complete valid five-arm suite")
    states = summary.get("arms")
    if not isinstance(states, list) or {state.get("id") for state in states} != {row["id"] for row in verified}:
        raise ValueError("summary.json arm set is incomplete")
    for state in states:
        if state.get("status") != "completed" or state.get("issues") or state.get("latency_valid") is not True:
            raise ValueError(f"summary.json contains an invalid arm: {state.get('id')}")
    required = [suite_dir / "summary.csv", suite_dir / "REPORT.md"] + [
        root / "assets" / SUITE / name
        for name in ("learning_curves.png", "comparison.png", "efficiency.png")
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"report output is missing/empty: {missing}")


def is_new_artifact(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in NEW_PREFIXES)


def verify_legacy_manifest(root: Path) -> list[dict[str, Any]]:
    manifest = read_object(root / "results" / "artifact_manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or manifest.get("artifact_count") != len(artifacts):
        raise ValueError("existing artifact manifest count is inconsistent")
    legacy = [item for item in artifacts if isinstance(item, dict) and not is_new_artifact(str(item.get("path", "")))]
    if len(legacy) != EXPECTED_LEGACY_ARTIFACTS:
        raise ValueError(f"expected {EXPECTED_LEGACY_ARTIFACTS} legacy artifacts, found {len(legacy)}")
    seen: set[str] = set()
    for item in legacy:
        relative = item.get("path")
        if not isinstance(relative, str) or relative in seen or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise ValueError(f"invalid legacy artifact path {relative!r}")
        seen.add(relative)
        path = root / Path(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"legacy artifact missing or not a regular file: {relative}")
        if path.stat().st_size != item.get("bytes") or sha256(path) != item.get("sha256"):
            raise ValueError(f"legacy artifact changed: {relative}")
    return legacy


def collect_new_artifacts(root: Path) -> list[dict[str, Any]]:
    files: list[Path] = []
    for prefix in NEW_PREFIXES:
        directory = root / prefix.rstrip("/")
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"artifact symlink is not allowed: {path}")
            if path.is_file() and not path.name.endswith(".tmp"):
                files.append(path)
    artifacts = []
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        artifacts.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return artifacts


def write_manifest(
    root: Path, legacy: list[dict[str, Any]], new: list[dict[str, Any]], *, will_stage: bool,
) -> None:
    artifacts = sorted([*legacy, *new], key=lambda item: item["path"])
    atomic_json(root / "results" / "artifact_manifest.json", {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "All tracked results, figures and checkpoints; excludes this manifest itself.",
        "suites": ["pilot", "module_extension", SUITE],
        "artifact_count": len(artifacts),
        "total_bytes": sum(item["bytes"] for item in artifacts),
        "git_staged_bytes_match_local_evidence": will_stage,
        "artifacts": artifacts,
    })


def readme_section(rows: list[dict[str, Any]]) -> str:
    lines = [
        README_START,
        "## Full-data pretrained real-time screening",
        "",
        "The five prespecified arms completed 100 epochs on all 6,471 training images and were selected by best validation AP50–95 on the fixed 548-image validation set.",
        "",
        "| Arm | Architecture | Input | Physical / effective batch | AP50–95 | AP50 | E2E p50 / p95 (ms) | FPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['id']}` | {row['architecture']} | {row['imgsz']} | "
            f"{row['physical_batch']} / {row['effective_batch']} | "
            f"{100 * row['map50_95']:.2f}% | {100 * row['map50']:.2f}% | "
            f"{row['latency_p50_ms']:.2f} / {row['latency_p95_ms']:.2f} | {row['latency_fps']:.1f} |"
        )
    lines += [
        "",
        "Training used the per-arm physical batches shown above and a common effective batch through gradient accumulation. Latency is synchronized end-to-end batch-1 FP16 inference on the recorded hardware, with a fused model graph, 30 warmups, and 100 timed images; FPS is derived from mean latency, while the table reports p50 and p95 latency. See the [complete report](results/realtime_stage1/REPORT.md), [machine-readable summary](results/realtime_stage1/summary.json), and [efficiency plot](assets/realtime_stage1/efficiency.png).",
        "",
        "These are single-seed screening results, so they do not estimate training variance. The full-data budget, COCO pretraining, and architecture/resolution changes were introduced jointly relative to the earlier pilot; their effect is not isolated. Evaluation uses the converted YOLO validation labels and does not implement the official VisDrone ignore-region matching rules.",
        README_END,
    ]
    return "\n".join(lines)


def update_readme(root: Path, rows: list[dict[str, Any]]) -> None:
    path = root / "README.md"
    text = path.read_text(encoding="utf-8")
    if text.count(README_START) != 1 or text.count(README_END) != 1:
        raise ValueError("README must contain exactly one realtime-stage1 marker pair")
    start, end = text.index(README_START), text.index(README_END)
    if start >= end:
        raise ValueError("README realtime-stage1 markers are out of order")
    end += len(README_END)
    replacement = readme_section(rows)
    atomic_text(path, text[:start] + replacement + text[end:])


def git(root: Path, *args: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-c", "core.excludesFile=NUL", *args],
        cwd=root, check=True,
        text=True, capture_output=capture,
    )


def changed_paths(root: Path, *, cached: bool = False) -> set[str]:
    args = ("diff", "--cached", "--name-only", "-z") if cached else ("status", "--porcelain=v1", "-z", "--untracked-files=all")
    output = git(root, *args).stdout
    if cached:
        return {name for name in output.split("\0") if name}
    paths: set[str] = set()
    entries = [entry for entry in output.split("\0") if entry]
    index = 0
    while index < len(entries):
        entry = entries[index]
        status, name = entry[:2], entry[3:]
        paths.add(name)
        if status[0] in "RC" and index + 1 < len(entries):
            index += 1
            paths.add(entries[index])
        index += 1
    return paths


def allowed_publish_path(path: str) -> bool:
    return path in PUBLISH_FILES or is_new_artifact(path)


def require_clean_scope(root: Path) -> None:
    disallowed = sorted(path for path in changed_paths(root) if not allowed_publish_path(path))
    if disallowed:
        raise RuntimeError(f"uncommitted files outside the publication allowlist: {disallowed}")
    cached_disallowed = sorted(path for path in changed_paths(root, cached=True) if not allowed_publish_path(path))
    if cached_disallowed:
        raise RuntimeError(f"staged files outside the publication allowlist: {cached_disallowed}")


def verify_origin(root: Path) -> None:
    raw = git(root, "remote", "get-url", "--push", "origin").stdout.strip()
    accepted = raw in {
        "https://github.com/DeepBluesL/VisDrone.git",
        "git@github.com:DeepBluesL/VisDrone.git",
        "ssh://git@github.com/DeepBluesL/VisDrone.git",
    }
    if raw.startswith("https://"):
        parsed = urlparse(raw)
        accepted = accepted and parsed.username is None and parsed.password is None
    if not accepted:
        raise RuntimeError("origin push URL is not the authorized DeepBluesL/VisDrone repository")


def publish(root: Path, *, commit: bool, push: bool) -> dict[str, Any]:
    if push and not commit:
        raise ValueError("--push requires --commit")
    if not commit:
        return {"committed": False, "pushed": False}
    require_clean_scope(root)
    git(root, "add", "--", "README.md", "results/artifact_manifest.json",
        f"results/{SUITE}", f"checkpoints/{SUITE}", f"assets/{SUITE}")
    cached = changed_paths(root, cached=True)
    disallowed = sorted(path for path in cached if not allowed_publish_path(path))
    if disallowed:
        raise RuntimeError(f"refusing to commit staged paths outside allowlist: {disallowed}")
    tracked = set(git(root, "ls-files", "-z").stdout.split("\0"))
    manifest = read_object(root / "results" / "artifact_manifest.json")
    missing_from_index = sorted(
        item["path"] for item in manifest["artifacts"] if item["path"] not in tracked
    )
    if missing_from_index:
        raise RuntimeError(f"manifest artifacts are absent from the Git index: {missing_from_index}")
    worktree_probe = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-c", "core.excludesFile=NUL",
         "diff", "--quiet", "--",
         "README.md", "results/artifact_manifest.json", f"results/{SUITE}",
         f"checkpoints/{SUITE}", f"assets/{SUITE}"], cwd=root,
    )
    if worktree_probe.returncode != 0:
        raise RuntimeError("staged publication bytes do not match local files")
    committed = False
    probe = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-c", "core.excludesFile=NUL",
         "diff", "--cached", "--quiet"], cwd=root
    )
    if probe.returncode not in (0, 1):
        raise RuntimeError("git diff --cached --quiet failed")
    if probe.returncode == 1:
        git(root, "commit", "-m", "Record realtime stage 1 results", capture=False)
        committed = True
    pushed = False
    if push:
        verify_origin(root)
        if git(root, "branch", "--show-current").stdout.strip() != "main":
            raise RuntimeError("publication push requires the main branch")
        git(root, "push", "origin", "main", capture=False)
        local = git(root, "rev-parse", "HEAD").stdout.strip()
        remote_line = git(root, "ls-remote", "origin", "refs/heads/main").stdout.strip()
        remote = remote_line.split()[0] if remote_line else ""
        if remote != local:
            raise RuntimeError("remote main SHA does not match local HEAD after push")
        pushed = True
    return {"committed": committed, "pushed": pushed}


def finalize(
    root: Path, *, commit: bool = False, push: bool = False, verify_only: bool = False,
) -> dict[str, Any]:
    if verify_only and (commit or push):
        raise ValueError("--verify-only cannot be combined with --commit or --push")
    require_clean_scope(root)
    protocol, rows = verify_suite(root)
    if verify_only:
        verify_report(root, rows)
        legacy = verify_legacy_manifest(root)
        return {
            "status": "verified", "suite": SUITE, "protocol_sha256": canonical_sha(protocol),
            "arms": len(rows), "legacy_artifacts": len(legacy),
        }
    subprocess.run(
        [sys.executable, str(root / "scripts" / "report_realtime.py"), "--suite", SUITE],
        cwd=root, check=True,
    )
    verify_report(root, rows)
    legacy = verify_legacy_manifest(root)
    update_readme(root, rows)
    new = collect_new_artifacts(root)
    write_manifest(root, legacy, new, will_stage=commit)
    result = publish(root, commit=commit, push=push)
    return {
        "status": "completed", "suite": SUITE, "protocol_sha256": canonical_sha(protocol),
        "arms": len(rows), "legacy_artifacts": len(legacy), "new_artifacts": len(new), **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--verify-only", action="store_true",
                        help="validate existing evidence without writing any file")
    args = parser.parse_args()
    if args.verify_only:
        result = finalize(ROOT, verify_only=True)
        print(json.dumps(result, ensure_ascii=False))
        return
    status_path = ROOT / ".runtime" / "finalization_state.json"
    started = time.time()
    atomic_json(status_path, {"status": "running", "suite": SUITE, "started": started})
    try:
        result = finalize(ROOT, commit=args.commit, push=args.push)
    except BaseException as exc:
        atomic_json(status_path, {
            "status": "failed", "suite": SUITE, "started": started,
            "finished": time.time(), "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    result.update(started=started, finished=time.time())
    atomic_json(status_path, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
