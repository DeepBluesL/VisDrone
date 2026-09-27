#!/usr/bin/env python3
"""Build an honest, protocol-driven report for the real-time optimization suite."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _protocol_arms(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    raw = protocol.get("arms")
    if not isinstance(raw, list) or not raw:
        raise ValueError("protocol.json must define a non-empty 'arms' list")
    defaults = protocol.get("training", {}) if isinstance(protocol.get("training"), dict) else {}
    arms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"protocol arm {index} must be an object")
        arm = {**defaults, **item}
        arm_id = str(_first(arm, "id", "run_id", "name") or "").strip()
        architecture = str(_first(arm, "architecture", "model_family") or "").strip()
        imgsz = _first(arm, "imgsz", "image_size")
        seed = arm.get("seed")
        if not arm_id or not architecture or not isinstance(imgsz, int) or not isinstance(seed, int):
            raise ValueError(f"arm {index} requires id, architecture, integer imgsz, and integer seed")
        if arm_id in seen or Path(arm_id).name != arm_id:
            raise ValueError(f"invalid or duplicate arm id: {arm_id!r}")
        seen.add(arm_id)
        arm.update(id=arm_id, architecture=architecture, imgsz=imgsz, seed=seed)
        if "epochs" not in arm and isinstance(protocol.get("epochs"), int):
            arm["epochs"] = protocol["epochs"]
        arms.append(arm)
    return arms


def _metric_values(record: dict[str, Any]) -> dict[str, float | None]:
    raw = record.get("metrics", {})
    if not isinstance(raw, dict):
        raw = {}
    return {name: _number(raw.get(key)) for name, key in METRICS.items()}


def _display_label(state: dict[str, Any], *, multiline: bool = False) -> str:
    """Keep homogeneous tiny-optimization arms distinguishable in standalone figures."""
    if "optimization" in state:
        return state["id"]
    separator = "\n" if multiline else " "
    return f"{state['architecture']}{separator}{state['imgsz']}" + (
        "" if multiline else f" (s{state['seed']})")


def _check_identity(arm: dict[str, Any], record: dict[str, Any], *, allow_missing: bool = False) -> list[str]:
    errors: list[str] = []
    required = (("seed", arm["seed"]), ("imgsz", arm["imgsz"]))
    if isinstance(arm.get("epochs"), int):
        required += (("epochs", arm["epochs"]),)
    for key, expected in required:
        actual = record.get(key)
        if actual is None and isinstance(record.get("training"), dict):
            actual = record["training"].get(key)
        if actual is None and allow_missing:
            continue
        if actual != expected:
            errors.append(f"{key}={actual!r}, expected {expected!r}")
    actual_arch = _first(record, "architecture", "model_family")
    if actual_arch is None and isinstance(record.get("training"), dict):
        actual_arch = _first(record["training"], "architecture", "model_family")
    # New records should expose architecture. Legacy-compatible records may identify the arm by variant.
    if actual_arch is not None and str(actual_arch) != arm["architecture"]:
        errors.append(f"architecture={actual_arch!r}, expected {arm['architecture']!r}")
    actual_id = _first(record, "arm_id", "run_id")
    if actual_id is None and record.get("variant") in (arm["id"], arm["architecture"]):
        actual_id = arm["id"]
    if actual_id is not None and str(actual_id) != arm["id"]:
        errors.append(f"arm_id={actual_id!r}, expected {arm['id']!r}")
    return errors


def _history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            epoch = _number(raw.get("epoch"))
            ap = _number(raw.get(METRICS["map50_95"]))
            ap50 = _number(raw.get(METRICS["map50"]))
            if epoch is not None and (ap is not None or ap50 is not None):
                # Ultralytics writes human-facing, one-based epoch numbers.
                rows.append({"epoch": epoch, "map50_95": ap, "map50": ap50})
    return rows


def _pause_evidence(suite_dir: Path, run_dir: Path, arm: dict[str, Any]) -> dict[str, Any] | None:
    progress_path = run_dir / "progress.json"
    progress = _read_json(progress_path) if progress_path.exists() else {}
    if progress.get("status") == "paused":
        return progress
    queue_path = suite_dir / "queue_state.json"
    if queue_path.exists():
        queue = _read_json(queue_path)
        if queue.get("status") == "paused" and queue.get("arm") == arm["id"]:
            return {**progress, "status": "paused", "epoch": queue.get("paused_epoch", progress.get("epoch", 0)),
                    "pause_reason": queue.get("reason", progress.get("pause_reason"))}
    return None


def _run_state(suite_dir: Path, arm: dict[str, Any]) -> dict[str, Any]:
    run_dir = suite_dir / arm["id"]
    metrics_path, progress_path = run_dir / "metrics.json", run_dir / "progress.json"
    state: dict[str, Any] = {"id": arm["id"], "architecture": arm["architecture"],
                             "imgsz": arm["imgsz"], "seed": arm["seed"],
                             "expected_epochs": arm.get("epochs"), "status": "pending",
                             "epoch": 0, "history": _history(run_dir / "epochs.csv"),
                             "issues": []}
    if "optimization" in arm:
        state["optimization"] = arm["optimization"]
    record: dict[str, Any] | None = None
    if metrics_path.exists():
        try:
            record = _read_json(metrics_path)
            state["issues"] = _check_identity(arm, record)
            protocol = _read_json(suite_dir / "protocol.json")
            expected_identity = {"protocol_sha256": _canonical_sha(protocol),
                                 "arm": next((item for item in protocol["arms"] if item["id"] == arm["id"]), None)}
            if record.get("experiment_identity") != expected_identity:
                state["issues"].append("experiment_identity does not match this protocol and arm")
            checkpoint = ROOT / "checkpoints" / suite_dir.name / f"{arm['id']}.pt"
            if not checkpoint.is_file():
                state["issues"].append("exported checkpoint is missing")
            elif record.get("checkpoint_sha256") != _sha256(checkpoint):
                state["issues"].append("exported checkpoint SHA-256 does not match metrics.json")
            values = _metric_values(record)
            if record.get("status") != "completed":
                state["issues"].append(f"metrics status is {record.get('status')!r}, expected 'completed'")
            if any(value is None for value in values.values()):
                state["issues"].append("one or more required detection metrics are missing/non-finite")
            state["status"] = "invalid" if state["issues"] else "completed"
            state["epoch"] = record.get("epochs", arm.get("epochs", 0))
            state.update(values)
            speed = record.get("speed_ms", {}) if isinstance(record.get("speed_ms"), dict) else {}
            state.update(parameters=record.get("parameters"), gflops_at_imgsz=record.get("gflops_at_imgsz"),
                         inference_ms=_number(speed.get("inference")), batch=record.get("batch"),
                         effective_batch=_first(record, "effective_batch", "effective_batch_size"),
                         selected_epoch=record.get("selected_epoch"), initialization=record.get("initialization"),
                         pretrained_coverage=_first(record, "pretrained_coverage", "pretrained_parameter_coverage"),
                         metrics_sha256=_sha256(metrics_path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.update(status="invalid", issues=[f"cannot read metrics.json: {exc}"])
    else:
        try:
            paused = _pause_evidence(suite_dir, run_dir, arm)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            paused = None
            state.update(status="invalid", issues=[f"cannot read pause state: {exc}"])
    if record is None and state["status"] != "invalid" and paused is not None:
        state["issues"] = _check_identity(arm, paused, allow_missing=True)
        state["status"] = "invalid" if state["issues"] else "paused"
        state["epoch"] = paused.get("epoch", 0)
        state.update(_metric_values(paused))
        if paused.get("pause_reason"):
            state["pause_reason"] = str(paused["pause_reason"])
    elif record is None and state["status"] != "invalid" and (run_dir / "failure.json").exists():
        try:
            failure = _read_json(run_dir / "failure.json")
            state["issues"] = _check_identity(arm, failure, allow_missing=True)
            state["status"] = "failed" if not state["issues"] else "invalid"
            state["epoch"] = failure.get("epoch", 0)
            if failure.get("error"):
                state["issues"].append(str(failure["error"]))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.update(status="invalid", issues=[f"cannot read failure.json: {exc}"])
    elif record is None and state["status"] != "invalid" and progress_path.exists():
        try:
            progress = _read_json(progress_path)
            state["issues"] = _check_identity(arm, progress, allow_missing=True)
            state["status"] = "invalid" if state["issues"] else "running"
            state["epoch"] = progress.get("epoch", 0)
            state.update(_metric_values(progress))
            queue_path = suite_dir / "queue_state.json"
            if state["status"] == "running" and queue_path.exists():
                queue = _read_json(queue_path)
                if queue.get("status") == "failed" and queue.get("arm") == arm["id"]:
                    state["status"] = "failed"
                    state["issues"].append(f"queue failed: {queue.get('error', 'unspecified error')}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.update(status="invalid", issues=[f"cannot read progress.json: {exc}"])
    if state["status"] == "completed":
        latency_path = run_dir / "latency.json"
        if latency_path.exists():
            try:
                latency = _read_json(latency_path)
                wall = latency.get("synchronized_wall_clock", {})
                expected = {
                    "schema_version": 1, "status": "completed", "arm_id": arm["id"],
                    "imgsz": arm["imgsz"], "seed": arm["seed"], "precision": "FP16",
                    "fused": True, "batch": 1,
                    "checkpoint_sha256": record.get("checkpoint_sha256") if record else None,
                    "metrics_json_sha256": state.get("metrics_sha256"),
                    "protocol_json_sha256": _sha256(suite_dir / "protocol.json"),
                    "warmup_iterations": 30, "timed_iterations": 100,
                    "settings": {"conf": 0.25, "iou": 0.7, "max_det": 500, "rect": False},
                }
                pure = latency.get("pure_model_cuda_event", {})
                valid = (all(latency.get(key) == value for key, value in expected.items())
                         and isinstance(pure, dict) and pure.get("samples") == 100
                         and wall.get("samples") == 100)
                values = {key: _number(wall.get(key)) for key in ("p50_ms", "p95_ms", "fps_from_mean")}
                if (valid and all(value is not None and value > 0 for value in values.values())
                        and values["p95_ms"] >= values["p50_ms"]):
                    state.update(latency_valid=True, latency_p50_ms=values["p50_ms"],
                                 latency_p95_ms=values["p95_ms"], latency_fps=values["fps_from_mean"],
                                 latency_sha256=_sha256(latency_path))
                else:
                    state["latency_valid"] = False
                    state["issues"].append("latency.json exists but provenance or summary values do not match")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                state["latency_valid"] = False
                state["issues"].append(f"cannot use latency.json: {exc}")
    return state


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_learning(states: list[dict[str, Any]], path: Path) -> bool:
    available = [state for state in states if state["history"]]
    if not available:
        path.unlink(missing_ok=True)
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(states), 2)))
    for color, state in zip(colors, states):
        if not state["history"]:
            continue
        epochs = [row["epoch"] for row in state["history"]]
        style = "-" if state["status"] == "completed" else "--"
        suffix = "" if state["status"] == "completed" else f" [{state['status']}]"
        label = _display_label(state) + suffix
        for axis, key, title in zip(axes, ("map50_95", "map50"), ("AP50–95", "AP50")):
            points = [(x, row[key]) for x, row in zip(epochs, state["history"]) if row[key] is not None]
            if points:
                axis.plot(*zip(*points), linestyle=style, color=color, linewidth=1.8, label=label)
            axis.set_title(title + " by epoch")
            axis.set_xlabel("Completed epoch")
            axis.set_ylabel(title)
            axis.grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(3, len(labels)), fontsize=8)
    fig.suptitle("Resolution × architecture learning curves (partial runs are dashed)")
    fig.subplots_adjust(bottom=.24, top=.86, wspace=.25)
    _save_figure(fig, path)
    return True


def _plot_comparison(completed: list[dict[str, Any]], path: Path) -> bool:
    if not completed:
        path.unlink(missing_ok=True)
        return False
    labels = [_display_label(s, multiline=True) for s in completed]
    x = np.arange(len(completed))
    width = .36
    fig, axis = plt.subplots(figsize=(max(8, len(completed) * 1.6), 5.2))
    axis.bar(x - width / 2, [s["map50_95"] for s in completed], width, label="AP50–95")
    axis.bar(x + width / 2, [s["map50"] for s in completed], width, label="AP50")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Validation AP")
    axis.set_title("Completed arms only — one seed per arm; no uncertainty estimate")
    axis.grid(axis="y", alpha=.25)
    axis.legend()
    _save_figure(fig, path)
    return True


def _plot_efficiency(completed: list[dict[str, Any]], path: Path) -> bool:
    usable = [s for s in completed if _number(s.get("parameters")) is not None
              and _number(s.get("gflops_at_imgsz")) is not None]
    if not usable:
        path.unlink(missing_ok=True)
        return False
    latency = [s for s in usable if s.get("latency_valid")]
    panels = 3 if latency else 2
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 5))
    for state in usable:
        label = _display_label(state)
        axes[0].scatter(float(state["parameters"]) / 1e6, state["map50_95"], s=60)
        axes[0].annotate(label, (float(state["parameters"]) / 1e6, state["map50_95"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=8)
        axes[1].scatter(float(state["gflops_at_imgsz"]), state["map50_95"], s=60)
        axes[1].annotate(label, (float(state["gflops_at_imgsz"]), state["map50_95"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=8)
    for axis, xlabel in zip(axes, ("Parameters (million)", "GFLOPs at the arm's input size")):
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Validation AP50–95")
        axis.grid(alpha=.25)
    if latency:
        for state in latency:
            label = f"{_display_label(state)}\n{state['latency_fps']:.1f} FPS"
            delta = max(0.0, state["latency_p95_ms"] - state["latency_p50_ms"])
            axes[2].errorbar(state["latency_p50_ms"], state["map50_95"], xerr=[[0.0], [delta]],
                             fmt="o", capsize=4)
            axes[2].annotate(label, (state["latency_p50_ms"], state["map50_95"]),
                             xytext=(4, 4), textcoords="offset points", fontsize=8)
        axes[2].set_xlabel("Synchronized batch-1 latency (p50→p95 ms)")
        axes[2].set_ylabel("Validation AP50–95")
        axes[2].grid(alpha=.25)
    fig.suptitle("Accuracy versus model size and compute — completed arms, one seed each")
    _save_figure(fig, path)
    return True


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _write_csv(states: list[dict[str, Any]], path: Path) -> None:
    fields = ["id", "architecture", "optimization", "imgsz", "seed", "status", "epoch", "expected_epochs",
              "precision", "recall", "map50", "map50_95", "parameters", "gflops_at_imgsz",
              "inference_ms", "latency_p50_ms", "latency_p95_ms", "latency_fps",
              "batch", "effective_batch", "selected_epoch", "issues"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for state in states:
            writer.writerow({key: ("; ".join(state.get(key, [])) if key == "issues"
                                   else json.dumps(state.get(key), sort_keys=True) if key == "optimization" and key in state
                                   else state.get(key)) for key in fields})
        temporary = Path(stream.name)
    temporary.replace(path)


def _report(suite: str, protocol: dict[str, Any], states: list[dict[str, Any]], plots: list[str]) -> str:
    counts = {name: sum(s["status"] == name for s in states)
              for name in ("completed", "running", "paused", "pending", "failed", "invalid")}
    has_optimization = any("optimization" in state for state in states)
    lines = [f"# {protocol.get('title', suite)}", "",
             f"Status: **{counts['completed']}/{len(states)} completed**, {counts['running']} running, "
             f"{counts['paused']} paused, {counts['pending']} pending, {counts['failed']} failed, {counts['invalid']} invalid.", "",
             "A paused arm is incomplete and is not counted as a completed result. It remains excluded from final accuracy and efficiency comparisons.", "",
             "Each row is one protocol arm and one seed. Incomplete arms are excluded from final "
             "accuracy and efficiency comparisons. No across-arm mean, standard deviation, confidence "
             "interval, or significance claim is reported.", ""]
    if has_optimization:
        lines += ["| Arm | Architecture | Optimization | Image size | Seed | Status | Epoch | Precision | Recall | AP50 | AP50–95 | Params | GFLOPs¹ | Validator ms² | E2E p50/p95 ms³ | FPS³ |",
                  "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    else:
        lines += ["| Arm | Architecture | Image size | Seed | Status | Epoch | Precision | Recall | AP50 | AP50–95 | Params | GFLOPs¹ | Validator ms² | E2E p50/p95 ms³ | FPS³ |",
                  "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in states:
        epoch = f"{s.get('epoch', 0)}/{s.get('expected_epochs') or '?'}"
        optimization = (json.dumps(s.get("optimization"), sort_keys=True, separators=(",", ":"))
                        if "optimization" in s else None)
        prefix = (f"| {s['id']} | {s['architecture']} | `{optimization}` | {s['imgsz']} | "
                  if has_optimization else f"| {s['id']} | {s['architecture']} | {s['imgsz']} | ")
        lines.append(prefix + f"{s['seed']} | {s['status']} | {epoch} | "
                     f"{_fmt(s.get('precision'))} | {_fmt(s.get('recall'))} | {_fmt(s.get('map50'))} | "
                     f"{_fmt(s.get('map50_95'))} | {_fmt(s.get('parameters'), 0)} | "
                     f"{_fmt(s.get('gflops_at_imgsz'), 3)} | {_fmt(s.get('inference_ms'), 3)} | "
                     f"{_fmt(s.get('latency_p50_ms'), 2)}/{_fmt(s.get('latency_p95_ms'), 2)} | "
                     f"{_fmt(s.get('latency_fps'), 1)} |")
    lines += ["", "¹ GFLOPs are shown at each arm's actual input size and are comparable only when the recorded accounting method agrees. "
              "² Validator inference is batched stage timing, not end-to-end latency. ³ E2E latency is the matched-provenance synchronized batch-1 benchmark; FPS is derived from mean latency, while the table shows p50/p95.", ""]
    if plots:
        lines += ["## Figures", ""] + [f"![{Path(name).stem.replace('_', ' ')}](../../assets/{suite}/{name})"
                                           for name in plots] + [""]
    invalid = [s for s in states if s["issues"]]
    if invalid:
        lines += ["## Validation issues", ""]
        for state in invalid:
            lines.append(f"- `{state['id']}`: " + "; ".join(state["issues"]))
        lines.append("")
    lines += ["## Interpretation limits", "",
              "The arms intentionally change input resolution and architecture. Compute, memory, and speed therefore belong to each arm's actual configuration. "
              "A single seed cannot quantify training variance. COCO-pretrained coverage may also differ for newly introduced P2 layers; consult each run's initialization and coverage metadata before attributing a difference solely to architecture.", "",
              "This validation uses the protocol's converted VisDrone validation set and is not the official VisDrone ignore-region evaluation unless the protocol explicitly says otherwise.", ""]
    return "\n".join(lines)


def generate(suite_dir: Path, assets_dir: Path) -> dict[str, Any]:
    protocol_path = suite_dir / "protocol.json"
    protocol = _read_json(protocol_path)
    arms = _protocol_arms(protocol)
    states = [_run_state(suite_dir, arm) for arm in arms]
    completed = [state for state in states if state["status"] == "completed"]
    assets_dir.mkdir(parents=True, exist_ok=True)
    plots: list[str] = []
    for filename, rendered in (
        ("learning_curves.png", _plot_learning(states, assets_dir / "learning_curves.png")),
        ("comparison.png", _plot_comparison(completed, assets_dir / "comparison.png")),
        ("efficiency.png", _plot_efficiency(completed, assets_dir / "efficiency.png")),
    ):
        if rendered:
            plots.append(filename)
    serializable = [{key: value for key, value in state.items() if key != "history"} for state in states]
    summary = {"schema_version": 1, "suite": suite_dir.name,
               "protocol_sha256": _sha256(protocol_path), "expected_arms": len(states),
               "completed_arms": len(completed),
               "status_counts": {name: sum(s["status"] == name for s in states)
                                 for name in ("completed", "running", "paused", "pending", "failed", "invalid")},
               "arms": serializable, "generated_plots": plots}
    _write_csv(serializable, suite_dir / "summary.csv")
    _atomic_json(suite_dir / "summary.json", summary)
    _atomic_text(suite_dir / "REPORT.md", _report(suite_dir.name, protocol, states, plots))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="realtime_stage1")
    args = parser.parse_args()
    if Path(args.suite).name != args.suite:
        raise SystemExit("--suite must be a single directory name")
    summary = generate(ROOT / "results" / args.suite, ROOT / "assets" / args.suite)
    print(json.dumps({"suite": summary["suite"], "completed": summary["completed_arms"],
                      "expected": summary["expected_arms"], "status": summary["status_counts"]}))


if __name__ == "__main__":
    main()
