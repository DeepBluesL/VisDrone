#!/usr/bin/env python3
"""Aggregate completed VisDrone experiment runs and build evidence-only reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FACTORS = ("c3k2", "ghost", "se", "sppf", "gabor")
SINGLE_ORDER = ("baseline",) + FACTORS
LEAVE_ONE_OUT_ORDER = ("full",) + tuple(f"without_{name}" for name in FACTORS)
VARIANT_ORDER = SINGLE_ORDER + LEAVE_ONE_OUT_ORDER + ("random_stem",)
DISPLAY_NAMES = {
    "baseline": "Baseline",
    "c3k2": "C3k2",
    "ghost": "Ghost",
    "se": "SE",
    "sppf": "Enhanced SPPF",
    "gabor": "Gabor stem",
    "full": "Full",
    "without_c3k2": "Full − C3k2",
    "without_ghost": "Full − Ghost",
    "without_se": "Full − SE",
    "without_sppf": "Full − SPPF",
    "without_gabor": "Full − Gabor",
    "random_stem": "Random stem control",
}
METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "ap50": "metrics/mAP50(B)",
    "ap50_95": "metrics/mAP50-95(B)",
}
CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
ENVIRONMENT_IDENTITY_KEYS = ("torch", "torchvision", "ultralytics", "numpy", "Pillow")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def _variant_sort_key(name: str) -> tuple[int, str]:
    try:
        return VARIANT_ORDER.index(name), name
    except ValueError:
        return len(VARIANT_ORDER), name


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def load_completed_runs(
    suite_dir: Path, root: Path = ROOT
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load and validate completed run records; non-completed records are ignored."""
    runs: list[dict[str, Any]] = []
    notices: list[str] = []
    for path in sorted(suite_dir.glob("*/metrics.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            notices.append(f"Skipped unreadable {_display_path(path, root)}: {exc}")
            continue
        if record.get("status") != "completed":
            notices.append(f"Skipped non-completed {_display_path(path, root)}")
            continue
        try:
            variant = str(record["variant"])
            seed = int(record["seed"])
            metrics = record["metrics"]
            normalized = {
                name: _finite_number(metrics[key], f"{path}:{key}")
                for name, key in METRIC_KEYS.items()
            }
        except (KeyError, TypeError, ValueError) as exc:
            notices.append(f"Skipped invalid completed record {_display_path(path, root)}: {exc}")
            continue
        record = dict(record)
        record["_path"] = path
        record["_run_id"] = path.parent.name
        record["_metrics"] = normalized
        record["variant"] = variant
        record["seed"] = seed
        runs.append(record)
    runs.sort(key=lambda run: (_variant_sort_key(run["variant"]), run["seed"], run["_run_id"]))
    return runs, notices


def _mean_std(values: Iterable[float]) -> tuple[float, float | None]:
    numbers = list(values)
    if not numbers:
        raise ValueError("cannot aggregate an empty value list")
    return statistics.fmean(numbers), statistics.stdev(numbers) if len(numbers) > 1 else None


def _optional_numbers(runs: Sequence[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        value = run.get(key)
        if value is not None:
            try:
                values.append(_finite_number(value, key))
            except ValueError:
                continue
    return values


def aggregate_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["variant"]].append(run)

    rows: list[dict[str, Any]] = []
    for variant in sorted(grouped, key=_variant_sort_key):
        group = grouped[variant]
        row: dict[str, Any] = {
            "variant": variant,
            "display_name": DISPLAY_NAMES.get(variant, variant),
            "run_count": len(group),
            "seeds": sorted(run["seed"] for run in group),
            "uncertainty": (
                "sample standard deviation across seeds"
                if len(group) > 1
                else "not estimated (one seed)"
            ),
        }
        for metric in METRIC_KEYS:
            mean, std = _mean_std(run["_metrics"][metric] * 100.0 for run in group)
            row[f"{metric}_percent_mean"] = mean
            row[f"{metric}_percent_std"] = std

        scalar_sources = {
            "parameters": "parameters",
            "gflops_at_imgsz": "gflops_at_imgsz",
            "training_seconds": "training_and_final_validation_seconds",
            "peak_cuda_reserved_gb": "peak_cuda_reserved_gb",
        }
        for output_key, source_key in scalar_sources.items():
            values = _optional_numbers(group, source_key)
            row[f"{output_key}_mean"] = statistics.fmean(values) if values else None
            row[f"{output_key}_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )

        inference = []
        for run in group:
            speed = run.get("speed_ms") or {}
            if speed.get("inference") is not None:
                try:
                    inference.append(_finite_number(speed["inference"], "speed_ms.inference"))
                except ValueError:
                    pass
        row["inference_ms_per_image_mean"] = statistics.fmean(inference) if inference else None
        row["inference_ms_per_image_std"] = (
            statistics.stdev(inference) if len(inference) > 1 else None
        )
        rows.append(row)
    return rows


def validate_suite_comparability(runs: Sequence[dict[str, Any]]) -> None:
    """Refuse to pool runs whose fixed protocol fields differ."""
    signatures: dict[str, set[str]] = defaultdict(set)
    direct_fields = (
        "epochs",
        "imgsz",
        "batch",
        "initialization",
        "evaluation_split",
        "evaluation_protocol",
        "checkpoint_selection",
        "data_yaml_sha256",
        "data_fingerprint",
        "training_code_sha256",
        "gflops_method",
    )
    for run in runs:
        missing = [
            field
            for field in (
                "data_fingerprint",
                "training_code_sha256",
                "environment_fingerprint",
            )
            if run.get(field) is None
        ]
        environment = run.get("environment_fingerprint") or {}
        packages = environment.get("packages") or {}
        if environment.get("python") is None:
            missing.append("environment_fingerprint.python")
        missing.extend(
            f"environment_fingerprint.packages.{key}"
            for key in ENVIRONMENT_IDENTITY_KEYS
            if packages.get(key) is None
        )
        if missing:
            raise ValueError(
                f"{run['_run_id']} lacks required provenance: {', '.join(missing)}"
            )
        for field in direct_fields:
            if field in run:
                signatures[field].add(json.dumps(run[field], sort_keys=True))
        environment_identity = {
            "python": environment["python"],
            "packages": {
                key: packages[key] for key in ENVIRONMENT_IDENTITY_KEYS
            },
        }
        signatures["core_environment"].add(
            json.dumps(environment_identity, sort_keys=True)
        )
        training = dict(run.get("training") or {})
        # These identify the run or execution target rather than its fixed
        # model and optimization protocol.
        for varying in ("name", "seed", "device", "exist_ok"):
            training.pop(varying, None)
        if training:
            signatures["training"].add(json.dumps(training, sort_keys=True))
    mismatches = [field for field, values in signatures.items() if len(values) > 1]
    if mismatches:
        raise ValueError(
            "completed runs use different fixed protocol values for: "
            + ", ".join(sorted(mismatches))
            + "; use separate suite names instead of pooling them"
        )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _summary_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "variant",
        "display_name",
        "run_count",
        "seeds",
        "uncertainty",
        "precision_percent_mean",
        "precision_percent_std",
        "recall_percent_mean",
        "recall_percent_std",
        "ap50_percent_mean",
        "ap50_percent_std",
        "ap50_95_percent_mean",
        "ap50_95_percent_std",
        "parameters_mean",
        "parameters_std",
        "gflops_at_imgsz_mean",
        "gflops_at_imgsz_std",
        "inference_ms_per_image_mean",
        "inference_ms_per_image_std",
        "training_seconds_mean",
        "training_seconds_std",
        "peak_cuda_reserved_gb_mean",
        "peak_cuda_reserved_gb_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field) for field in fields}
            output["seeds"] = ";".join(str(seed) for seed in row["seeds"])
            writer.writerow(output)
    temporary.replace(path)


def _style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _errorbar_for_bar(ax: plt.Axes, x: float, y: float, std: float | None) -> None:
    if std is not None:
        ax.errorbar(x, y, yerr=std, color="#222222", capsize=3, linewidth=1.2)


def plot_comparison(rows: Sequence[dict[str, Any]], path: Path) -> bool:
    by_variant = {row["variant"]: row for row in rows}
    selected = [by_variant[name] for name in SINGLE_ORDER if name in by_variant]
    if not selected:
        return False
    x = list(range(len(selected)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ap50 = [row["ap50_percent_mean"] for row in selected]
    ap = [row["ap50_95_percent_mean"] for row in selected]
    ax.bar([value - width / 2 for value in x], ap50, width, label="AP50 (%)", color="#4C78A8")
    ax.bar([value + width / 2 for value in x], ap, width, label="AP50–95 (%)", color="#F58518")
    for position, row in zip(x, selected):
        _errorbar_for_bar(ax, position - width / 2, row["ap50_percent_mean"], row["ap50_percent_std"])
        _errorbar_for_bar(ax, position + width / 2, row["ap50_95_percent_mean"], row["ap50_95_percent_std"])
    ax.set_xticks(x, [row["display_name"] for row in selected], rotation=18, ha="right")
    _style_axis(ax, "Average precision (%)")
    ax.set_title("Baseline vs. single-module interventions")
    ax.legend(frameon=False)
    _save_figure(fig, path)
    return True


def plot_ablation(rows: Sequence[dict[str, Any]], path: Path) -> bool:
    by_variant = {row["variant"]: row for row in rows}
    ablation = [by_variant[name] for name in LEAVE_ONE_OUT_ORDER if name in by_variant]
    control_names = ("baseline", "gabor", "random_stem")
    controls = [by_variant[name] for name in control_names if name in by_variant]
    if not ablation and not controls:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    panels = (
        (axes[0], ablation, "Full model and leave-one-out ablations", "#54A24B"),
        (axes[1], controls, "Fixed Gabor stem and random-filter control", "#B279A2"),
    )
    for ax, selected, title, color in panels:
        positions = list(range(len(selected)))
        values = [row["ap50_95_percent_mean"] for row in selected]
        ax.bar(positions, values, color=color, alpha=0.9)
        for position, row in zip(positions, selected):
            _errorbar_for_bar(ax, position, row["ap50_95_percent_mean"], row["ap50_95_percent_std"])
        ax.set_xticks(positions, [row["display_name"] for row in selected], rotation=25, ha="right")
        _style_axis(ax, "AP50–95 (%)")
        ax.set_title(title)
        if not selected:
            ax.text(0.5, 0.5, "No completed runs", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    _save_figure(fig, path)
    return True


def _load_epoch_histories(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    for run in runs:
        path = run["_path"].with_name("epochs.csv")
        if not path.is_file():
            continue
        points: dict[int, float] = {}
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                for raw_row in reader:
                    row = {(key or "").strip(): value for key, value in raw_row.items()}
                    if "epoch" not in row or METRIC_KEYS["ap50_95"] not in row:
                        continue
                    # Ultralytics results.csv stores human-facing epochs starting at 1.
                    epoch = int(float(row["epoch"]))
                    points[epoch] = _finite_number(row[METRIC_KEYS["ap50_95"]], str(path)) * 100.0
        except (OSError, ValueError):
            continue
        if points:
            histories.append({"variant": run["variant"], "seed": run["seed"], "points": points})
    return histories


def plot_learning_curves(runs: Sequence[dict[str, Any]], path: Path) -> bool:
    histories = _load_epoch_histories(runs)
    if not histories:
        return False
    grouped: dict[str, list[dict[int, float]]] = defaultdict(list)
    for history in histories:
        grouped[history["variant"]].append(history["points"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    panel_orders = (SINGLE_ORDER, LEAVE_ONE_OUT_ORDER + ("random_stem",))
    titles = ("Baseline and single interventions", "Full model, ablations, and control")
    palette = plt.get_cmap("tab10")
    drew_any = False
    for ax, order, title in zip(axes, panel_orders, titles):
        for color_index, variant in enumerate(order):
            series = grouped.get(variant, [])
            if not series:
                continue
            epochs = sorted(set().union(*(points.keys() for points in series)))
            means: list[float] = []
            stds: list[float | None] = []
            for epoch in epochs:
                values = [points[epoch] for points in series if epoch in points]
                mean, std = _mean_std(values)
                means.append(mean)
                stds.append(std)
            color = palette(color_index % 10)
            ax.plot(epochs, means, label=DISPLAY_NAMES.get(variant, variant), color=color, linewidth=1.8)
            if any(value is not None for value in stds):
                lower = [mean - (std or 0.0) for mean, std in zip(means, stds)]
                upper = [mean + (std or 0.0) for mean, std in zip(means, stds)]
                ax.fill_between(epochs, lower, upper, color=color, alpha=0.14)
            drew_any = True
        ax.set_xlabel("Epoch")
        _style_axis(ax, "Validation AP50–95 (%)")
        ax.set_title(title)
        if ax.lines:
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No completed runs yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
    fig.tight_layout()
    if drew_any:
        _save_figure(fig, path)
    else:
        plt.close(fig)
    return drew_any


def _aggregate_per_class(runs: Sequence[dict[str, Any]]) -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for item in run.get("per_class") or []:
            try:
                class_id = int(item["id"])
                value = _finite_number(item["ap50_95"], "per_class.ap50_95") * 100.0
            except (KeyError, TypeError, ValueError):
                continue
            grouped[run["variant"]][class_id].append(value)
    return {
        variant: {class_id: statistics.fmean(values) for class_id, values in classes.items()}
        for variant, classes in grouped.items()
    }


def plot_per_class(runs: Sequence[dict[str, Any]], path: Path) -> bool:
    values = _aggregate_per_class(runs)
    variants = [name for name in VARIANT_ORDER if name in values]
    variants.extend(sorted(set(values) - set(variants)))
    if not variants:
        return False
    matrix = [[values[variant].get(index, float("nan")) for index in range(10)] for variant in variants]
    height = max(5.0, 0.42 * len(variants) + 2.2)
    fig, ax = plt.subplots(figsize=(13.5, height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(10), CLASS_NAMES, rotation=28, ha="right")
    ax.set_yticks(range(len(variants)), [DISPLAY_NAMES.get(name, name) for name in variants])
    ax.set_title("Per-class validation AP50–95 (%)")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("AP50–95 (%)")
    fig.tight_layout()
    _save_figure(fig, path)
    return True


def plot_efficiency(rows: Sequence[dict[str, Any]], path: Path) -> bool:
    usable = [
        row
        for row in rows
        if row.get("parameters_mean") is not None and row.get("gflops_at_imgsz_mean") is not None
    ]
    if not usable:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.0))
    legend_handles = []
    label_offsets = {
        "parameters_mean": {
            "ghost": (12, 13),
            "without_gabor": (-14, -14),
            "full": (-14, -13),
            "without_sppf": (13, 13),
        },
        "gflops_at_imgsz_mean": {
            "full": (-14, -13),
            "without_sppf": (13, 13),
        },
    }
    for ax, key, xlabel in (
        (axes[0], "parameters_mean", "Parameters (millions)"),
        (
            axes[1],
            "gflops_at_imgsz_mean",
            "Conv/Linear compute (GFLOPs, MACs × 2)",
        ),
    ):
        for index, row in enumerate(usable):
            x = row[key] / 1e6 if key == "parameters_mean" else row[key]
            y = row["ap50_95_percent_mean"]
            color = plt.get_cmap("tab10")(index % 10)
            point = ax.scatter(x, y, s=48, color=color, edgecolor="white", linewidth=0.6)
            offset = label_offsets[key].get(row["variant"], (7, 8))
            ax.annotate(
                str(index + 1),
                xy=(x, y),
                xytext=offset,
                textcoords="offset points",
                color="#222222",
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 0.7,
                    "alpha": 0.95,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "linewidth": 0.7,
                    "shrinkA": 1,
                    "shrinkB": 2,
                },
            )
            if ax is axes[0]:
                legend_handles.append(point)
        ax.set_xlabel(xlabel)
        _style_axis(ax, "AP50–95 (%)")
    axes[0].set_title("Accuracy vs. model size")
    axes[1].set_title("Accuracy vs. computation")
    fig.legend(
        legend_handles,
        [f"{index + 1}. {row['display_name']}" for index, row in enumerate(usable)],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.23, 1, 1))
    _save_figure(fig, path)
    return True


def load_occlusion_results(
    runs: Sequence[dict[str, Any]], root: Path = ROOT
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load optional per-run natural-occlusion diagnostics with provenance checks."""
    records: list[dict[str, Any]] = []
    notices: list[str] = []
    expected_settings = {
        "confidence_threshold": 0.05,
        "nms_iou_threshold": 0.5,
        "matching_iou_threshold": 0.5,
    }
    for run in runs:
        path = run["_path"].with_name("occlusion.json")
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            settings = payload["settings"]
            provenance = payload["provenance"]
            occlusion = payload["metrics"]["occlusion"]
            checkpoint_hash = run.get("checkpoint_sha256")
            if not checkpoint_hash:
                raise ValueError("run record has no checkpoint_sha256")
            if provenance.get("weights_sha256") != checkpoint_hash:
                raise ValueError("weights SHA-256 does not match the run checkpoint")
            for key, expected in expected_settings.items():
                actual = _finite_number(settings[key], f"{path}:{key}")
                if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"{key}={actual} (required {expected})")
            for key in (
                "source_image_count",
                "evaluated_image_count",
                "is_subset_smoke_run",
                "size_bins",
            ):
                if settings.get(key) is None:
                    raise ValueError(f"missing settings.{key}")
            if not provenance.get("filenames_sha256"):
                raise ValueError("missing provenance.filenames_sha256")
            if int(settings["evaluated_image_count"]) < 1:
                raise ValueError("evaluated_image_count must be positive")

            strata: dict[str, dict[str, int | float | None]] = {}
            for level in ("0", "1", "2"):
                item = occlusion[level]
                total = int(item["total"])
                matched = int(item["match_count"])
                if total < 0 or matched < 0 or matched > total:
                    raise ValueError(f"invalid counts for occlusion {level}")
                recall = item.get("recall")
                if total:
                    recall_value = _finite_number(recall, f"{path}:occlusion.{level}.recall")
                    expected_recall = matched / total
                    if not math.isclose(recall_value, expected_recall, rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError(f"recall/count mismatch for occlusion {level}")
                elif recall is not None:
                    raise ValueError(f"non-null recall with zero count for occlusion {level}")
                else:
                    recall_value = None
                strata[level] = {
                    "total": total,
                    "match_count": matched,
                    "recall": recall_value,
                }
            record = {
                "run_id": run["_run_id"],
                "variant": run["variant"],
                "seed": run["seed"],
                "path": path,
                "sha256": sha256(path),
                "settings": settings,
                "provenance": provenance,
                "strata": strata,
            }
            records.append(record)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            notices.append(f"Skipped invalid {_display_path(path, root)}: {exc}")

    if records:
        comparable_fields = {
            "evaluated filenames": {
                str(record["provenance"].get("filenames_sha256")) for record in records
            },
            "evaluated image count": {
                str(record["settings"].get("evaluated_image_count")) for record in records
            },
            "subset status": {
                str(record["settings"].get("is_subset_smoke_run")) for record in records
            },
            "size definition": {
                str(record["settings"].get("size_bins")) for record in records
            },
            "occlusion GT counts": {
                json.dumps(
                    {level: record["strata"][level]["total"] for level in ("0", "1", "2")},
                    sort_keys=True,
                )
                for record in records
            },
        }
        mismatches = [name for name, values in comparable_fields.items() if len(values) > 1]
        if mismatches:
            raise ValueError(
                "occlusion diagnostics are not comparable for: "
                + ", ".join(mismatches)
            )
    records.sort(
        key=lambda record: (
            _variant_sort_key(record["variant"]),
            record["seed"],
            record["run_id"],
        )
    )
    return records, notices


def aggregate_occlusion(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["variant"]].append(record)
    rows: list[dict[str, Any]] = []
    for variant in sorted(grouped, key=_variant_sort_key):
        group = grouped[variant]
        row: dict[str, Any] = {
            "variant": variant,
            "display_name": DISPLAY_NAMES.get(variant, variant),
            "run_count": len(group),
            "seeds": sorted(record["seed"] for record in group),
            "evaluated_image_count": group[0]["settings"].get("evaluated_image_count"),
            "is_subset_smoke_run": group[0]["settings"].get("is_subset_smoke_run"),
        }
        for level in ("0", "1", "2"):
            values = [
                record["strata"][level]["recall"] * 100.0
                for record in group
                if record["strata"][level]["recall"] is not None
            ]
            mean, std = _mean_std(values) if values else (None, None)
            row[f"occlusion_{level}_count"] = group[0]["strata"][level]["total"]
            row[f"occlusion_{level}_recall_percent_mean"] = mean
            row[f"occlusion_{level}_recall_percent_std"] = std
        rows.append(row)
    return rows


def plot_occlusion_recall(
    rows: Sequence[dict[str, Any]], path: Path
) -> bool:
    if not rows:
        return False
    x = list(range(len(rows)))
    width = 0.24
    offsets = (-width, 0.0, width)
    colors = ("#4C78A8", "#F2CF5B", "#E45756")
    fig_width = max(10.5, len(rows) * 0.72 + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    for level, offset, color in zip(("0", "1", "2"), offsets, colors):
        values = [row[f"occlusion_{level}_recall_percent_mean"] for row in rows]
        if all(value is None for value in values):
            continue
        plotted = [value if value is not None else 0.0 for value in values]
        count = rows[0][f"occlusion_{level}_count"]
        ax.bar(
            [position + offset for position in x],
            plotted,
            width,
            color=color,
            label=f"Occlusion {level} (GT n={count})",
        )
        for position, row, value in zip(x, rows, values):
            if value is not None:
                _errorbar_for_bar(
                    ax,
                    position + offset,
                    value,
                    row[f"occlusion_{level}_recall_percent_std"],
                )
    ax.set_xticks(x, [row["display_name"] for row in rows], rotation=25, ha="right")
    _style_axis(ax, "Class-aware ground-truth recall (%)")
    ax.set_title("Natural VisDrone occlusion recall (confidence 0.05, match IoU 0.50)")
    ax.legend(frameon=False)
    _save_figure(fig, path)
    return True


def _data_manifest_metadata(path: Path, root: Path = ROOT) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {"path": _display_path(path, root), "sha256": sha256(path), "content": content}


def _load_protocol(path: Path, root: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable suite protocol: {exc}") from exc
    return {
        "path": _display_path(path, root),
        "sha256": sha256(path),
        "content": content,
    }


def validate_protocol_provenance(
    protocol: dict[str, Any] | None,
    runs: Sequence[dict[str, Any]],
    manifest: dict[str, Any] | None,
) -> None:
    if protocol is None:
        return
    content = protocol["content"]
    first = runs[0]
    for key in ("data_fingerprint", "training_code_sha256", "environment_fingerprint"):
        if content.get(key) != first.get(key):
            raise ValueError(f"protocol {key} does not match completed runs")
    expected_manifest_hash = content.get("preparation_manifest_sha256")
    if manifest is not None and expected_manifest_hash != manifest["sha256"]:
        raise ValueError("suite preparation-manifest snapshot SHA-256 does not match protocol")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _metric_cell(row: dict[str, Any], metric: str) -> str:
    mean = row[f"{metric}_percent_mean"]
    std = row[f"{metric}_percent_std"]
    return f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f}"


def build_markdown_report(
    suite: str,
    runs: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any] | None,
    occlusion_rows: Sequence[dict[str, Any]],
    generated_plots: Sequence[str],
    notices: Sequence[str],
) -> str:
    completed_variants = {
        run["variant"] for run in runs if run["variant"] in VARIANT_ORDER
    }
    if completed_variants == set(VARIANT_ORDER):
        suite_status = (
            f"> Suite 状态：{len(VARIANT_ORDER)}/{len(VARIANT_ORDER)} 个预定义变体均已完成。"
        )
    else:
        suite_status = (
            f"> Suite 状态：已完成 {len(completed_variants)}/{len(VARIANT_ORDER)} 个预定义变体。"
            "结果仍不完整；下表和图只反映当前已完成运行。"
        )
    lines = [
        f"# VisDrone 检测实验报告：{suite}",
        "",
        "本报告只汇总 `status=completed` 的真实运行记录。AP 数值均为百分数。这里的验证是由 VisDrone 标注转换得到的标准 YOLO 10 类评估；它没有实现官方 VisDrone 评测器对忽略区域的匹配规则，因此不能作为官方 VisDrone DET 榜单成绩。",
        "",
        suite_status,
        "",
        "## 汇总结果",
        "",
        "| 变体 | 运行数 | 种子 | Precision (%) | Recall (%) | AP50 (%) | AP50–95 (%) | 参数量 | Conv/Linear GFLOPs¹ | 验证器 inference (ms/图)² |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        params = row["parameters_mean"]
        params_text = f"{params / 1e6:.3f}M" if params is not None else "—"
        lines.append(
            "| {name} | {count} | {seeds} | {precision} | {recall} | {ap50} | {ap} | {params} | {gflops} | {latency} |".format(
                name=row["display_name"],
                count=row["run_count"],
                seeds=", ".join(str(seed) for seed in row["seeds"]),
                precision=_metric_cell(row, "precision"),
                recall=_metric_cell(row, "recall"),
                ap50=_metric_cell(row, "ap50"),
                ap=_metric_cell(row, "ap50_95"),
                params=params_text,
                gflops=_fmt(row["gflops_at_imgsz_mean"]),
                latency=_fmt(row["inference_ms_per_image_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "表中的 `±` 是多个种子之间的样本标准差。只有一个种子的变体仅报告单次观测值，没有不确定性估计，不能把它解释为稳定的模型差异。",
            "",
            "¹ GFLOPs 按 batch 1 的卷积和线性层 MACs×2 计算，包含固定 Gabor 卷积，不包含 BN、池化、激活、解码和 NMS。² inference 是 Ultralytics 验证器在批处理验证过程记录的每图阶段时间，不是隔离环境下的单图端到端延迟。",
            "",
            "## 实际运行协议",
            "",
        ]
    )
    first = runs[0]
    actual_fields = (
        ("Epoch 预算", first.get("epochs")),
        ("输入尺寸", first.get("imgsz")),
        ("Batch", first.get("batch")),
        ("初始化", first.get("initialization")),
        ("评估划分", first.get("evaluation_split")),
        ("Checkpoint 选择", first.get("checkpoint_selection")),
    )
    lines.extend(["| 项目 | 运行记录值 |", "|---|---|"])
    for label, value in actual_fields:
        if value is not None:
            lines.append(f"| {label} | {value} |")
    lines.extend(
        [
            "",
            "优化器与增强等完整实际参数保存在各运行的 `metrics.json`；suite 的固定调用参数保存在 `protocol.json`。",
            "",
            "## 数据与可复现性",
            "",
            "类别（源标注 1–10 映射到 YOLO 0–9）：" + "、".join(CLASS_NAMES) + "。",
            "",
        ]
    )
    if manifest is None:
        lines.append(
            "该 suite 未保存可读取的 `results/<suite>/data_snapshot/preparation_manifest.json` 数据快照，"
            "因此没有写入数据计数或清单哈希；报告不会用当前 `data/` 下可能已经变化的清单代替。"
        )
    else:
        content = manifest["content"]
        lines.append(f"数据准备清单 SHA-256：`{manifest['sha256']}`。")
        lines.append("")
        lines.append("| 划分 | 原始划分 | 原图数 | 使用图数 | 保留框 | 丢弃行 | 裁剪框 | 文件列表 SHA-256 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for split in ("train", "val"):
            item = (content.get("splits") or {}).get(split)
            if not item:
                continue
            stats = item.get("stats") or {}
            annotations = stats.get("annotation_rows") or {}
            lines.append(
                f"| {split} | {item.get('source_split', '—')} | {item.get('source_image_count', '—')} | "
                f"{item.get('selected_image_count', '—')} | {annotations.get('kept', '—')} | "
                f"{annotations.get('dropped', '—')} | {stats.get('clipped_boxes', '—')} | "
                f"`{item.get('yolo_list_sha256', '—')}` |"
            )
    data_hashes = sorted({str(run.get("data_yaml_sha256")) for run in runs if run.get("data_yaml_sha256")})
    lines.extend(["", "运行记录中的 dataset YAML SHA-256：" + ("、".join(f"`{value}`" for value in data_hashes) if data_hashes else "未记录。"), ""])
    lines.append(f"训练代码 SHA-256：`{first['training_code_sha256']}`。")
    lines.append("")
    lines.append("数据指纹（图像数及图像+标签 SHA-256）：")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(first["data_fingerprint"], indent=2, sort_keys=True))
    lines.append("```")
    environment_identity = first["environment_fingerprint"]
    lines.append("")
    lines.append("核心软件环境：`" + json.dumps(environment_identity, sort_keys=True) + "`。")
    lines.append("")
    lines.extend(["### 运行产物哈希", "", "| 运行 | metrics.json SHA-256 | checkpoint SHA-256 |", "|---|---|---|"])
    for run in runs:
        lines.append(f"| {run['_run_id']} | `{sha256(run['_path'])}` | `{run.get('checkpoint_sha256', '—')}` |")

    if occlusion_rows:
        lines.extend(
            [
                "",
                "## 自然遮挡召回诊断",
                "",
                "遮挡等级来自原始 VisDrone 标注字段。预测置信度阈值为 0.05，NMS IoU 为 0.50；按置信度降序进行类别一致、一对一的 IoU≥0.50 匹配，再按遮挡等级分层。下表是 ground-truth recall，不是 AP，也没有实现官方忽略区域匹配。`n` 是各层合格 GT 数；边界框大小如需分层，按原图像素面积计算（small < 32²、medium < 96²、large ≥ 96²）。",
                "",
                "| 变体 | 诊断运行数 | 图像数 | 遮挡 0 Recall (%) | 遮挡 1 Recall (%) | 遮挡 2 Recall (%) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in occlusion_rows:
            cells = []
            for level in ("0", "1", "2"):
                mean = row[f"occlusion_{level}_recall_percent_mean"]
                std = row[f"occlusion_{level}_recall_percent_std"]
                count = row[f"occlusion_{level}_count"]
                value = "—" if mean is None else (f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f}")
                cells.append(f"{value} (n={count})")
            image_count = row.get("evaluated_image_count", "—")
            if row.get("is_subset_smoke_run"):
                image_count = f"{image_count}（subset smoke）"
            lines.append(
                f"| {row['display_name']} | {row['run_count']} | {image_count} | "
                + " | ".join(cells)
                + " |"
            )

    if generated_plots:
        lines.extend(["", "## 图表", ""])
        captions = {
            "comparison.png": "基线与各单模块比较",
            "learning_curves.png": "验证集 AP50–95 学习曲线",
            "ablation.png": "完整模型、留一消融与随机滤波器对照",
            "per_class_ap.png": "各类别 AP50–95",
            "efficiency.png": "精度与参数量/计算量关系",
            "occlusion_recall.png": "自然遮挡等级召回率",
        }
        for filename in generated_plots:
            lines.extend([f"### {captions[filename]}", "", f"![{captions[filename]}](../../assets/{suite}/{filename})", ""])

    expected = set(VARIANT_ORDER)
    present = {row["variant"] for row in rows}
    missing = sorted(expected - present, key=_variant_sort_key)
    lines.extend(["## 解释限制", ""])
    if missing:
        lines.append("尚无完成结果的预定义变体：" + "、".join(DISPLAY_NAMES[name] for name in missing) + "。缺失比较不会被推断或补值。")
        lines.append("")
    lines.append("所有模型均从随机初始化开始，并在固定但较短的 epoch 预算内训练。这样的试验适合验证代码路径和比较早期学习行为，不能证明模型已经收敛，也不能据此下结论说某个结构在充分训练后一定更优。")
    lines.append("")
    lines.append("单模块试验回答“在基线上加入一个模块”的问题；留一消融回答“从完整组合中移除一个模块”的问题。两者的参照不同。`random_stem` 只用于检验固定 Gabor 滤波器相对同形状随机固定滤波器的作用，单独展示。")
    if notices:
        lines.extend(["", "## 未纳入的记录", ""])
        lines.extend(f"- {notice}" for notice in notices)
    return "\n".join(lines) + "\n"


def generate_report(root: Path, suite: str) -> int:
    suite_dir = root / "results" / suite
    runs, notices = load_completed_runs(suite_dir, root)
    for notice in notices:
        print(f"NOTICE: {notice}", file=sys.stderr)
    if not runs:
        print(f"No completed metrics.json records found under {suite_dir}; no report artifacts created.")
        return 0

    try:
        validate_suite_comparability(runs)
    except ValueError as exc:
        print(f"Cannot aggregate suite: {exc}", file=sys.stderr)
        return 2

    try:
        occlusion_records, occlusion_notices = load_occlusion_results(runs, root)
    except ValueError as exc:
        print(f"Cannot aggregate occlusion diagnostics: {exc}", file=sys.stderr)
        return 2
    notices.extend(occlusion_notices)
    for notice in occlusion_notices:
        print(f"NOTICE: {notice}", file=sys.stderr)

    rows = aggregate_runs(runs)
    occlusion_rows = aggregate_occlusion(occlusion_records)
    manifest = _data_manifest_metadata(
        suite_dir / "data_snapshot" / "preparation_manifest.json", root
    )
    protocol_path = suite_dir / "protocol.json"
    try:
        protocol = _load_protocol(protocol_path, root)
        validate_protocol_provenance(protocol, runs, manifest)
    except ValueError as exc:
        print(f"Cannot verify suite provenance: {exc}", file=sys.stderr)
        return 2
    if protocol is None:
        notice = "Suite protocol.json is missing; protocol snapshot provenance is unavailable."
        notices.append(notice)
        print(f"NOTICE: {notice}", file=sys.stderr)
    if manifest is None:
        notice = "Suite data_snapshot/preparation_manifest.json is missing; current data manifest was not substituted."
        notices.append(notice)
        print(f"NOTICE: {notice}", file=sys.stderr)

    assets_dir = root / "assets" / suite
    plotters = (
        ("comparison.png", lambda path: plot_comparison(rows, path)),
        ("learning_curves.png", lambda path: plot_learning_curves(runs, path)),
        ("ablation.png", lambda path: plot_ablation(rows, path)),
        ("per_class_ap.png", lambda path: plot_per_class(runs, path)),
        ("efficiency.png", lambda path: plot_efficiency(rows, path)),
        (
            "occlusion_recall.png",
            lambda path: plot_occlusion_recall(occlusion_rows, path),
        ),
    )
    generated_plots: list[str] = []
    for filename, make in plotters:
        plot_path = assets_dir / filename
        if make(plot_path):
            generated_plots.append(filename)
        elif plot_path.is_file():
            # Prevent an older chart from appearing to belong to the current
            # evidence set when its source fields are now absent.
            plot_path.unlink()

    summary = {
        "schema_version": 1,
        "suite": suite,
        "metric_units": {
            "precision": "percent",
            "recall": "percent",
            "ap50": "percent",
            "ap50_95": "percent",
        },
        "gflops_methods": sorted(
            {str(run["gflops_method"]) for run in runs if run.get("gflops_method")}
        ),
        "inference_timing_definition": (
            "Ultralytics batched validator speed_ms.inference per image; "
            "not isolated end-to-end single-image latency"
        ),
        "completed_run_count": len(runs),
        "aggregates": rows,
        "occlusion_diagnostic": {
            "units": "recall percent",
            "confidence_threshold": 0.05,
            "nms_iou_threshold": 0.5,
            "matching_iou_threshold": 0.5,
            "aggregates": occlusion_rows,
            "sources": [
                {
                    "run_id": record["run_id"],
                    "path": record["path"].relative_to(root).as_posix(),
                    "sha256": record["sha256"],
                    "weights_sha256": record["provenance"].get("weights_sha256"),
                    "filenames_sha256": record["provenance"].get("filenames_sha256"),
                }
                for record in occlusion_records
            ],
        },
        "source_metrics": [
            {
                "run_id": run["_run_id"],
                "path": run["_path"].relative_to(root).as_posix(),
                "sha256": sha256(run["_path"]),
                "checkpoint_sha256": run.get("checkpoint_sha256"),
            }
            for run in runs
        ],
        "protocol": protocol,
        "data_preparation_manifest": (
            {key: value for key, value in manifest.items() if key != "content"}
            if manifest
            else None
        ),
        "generated_plots": generated_plots,
        "notices": notices,
    }
    _summary_csv(rows, suite_dir / "summary.csv")
    _atomic_text(suite_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = build_markdown_report(
        suite,
        runs,
        rows,
        manifest,
        occlusion_rows,
        generated_plots,
        notices,
    )
    _atomic_text(suite_dir / "REPORT.md", report)
    print(f"Aggregated {len(runs)} completed runs into {suite_dir}")
    if len(generated_plots) < len(plotters):
        missing = sorted({name for name, _ in plotters} - set(generated_plots))
        print("Plots skipped because required real fields were absent: " + ", ".join(missing))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="pilot", help="suite directory name under results/")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite_path = Path(args.suite)
    if suite_path.name != args.suite or args.suite in {"", ".", ".."}:
        raise SystemExit("--suite must be one directory name, without path separators")
    return generate_report(ROOT, args.suite)


if __name__ == "__main__":
    raise SystemExit(main())
