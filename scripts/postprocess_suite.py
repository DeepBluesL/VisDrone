#!/usr/bin/env python3
"""Verify and postprocess every completed run declared by an experiment suite."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

from scripts.verify_results import verify_suite


CONFIDENCE = 0.05
NMS_IOU = 0.5
MATCH_IOU = 0.5
BATCH = 16
MAX_DET = 500


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _same_number(actual: Any, expected: float) -> bool:
    try:
        return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def reusable_occlusion(
    path: Path,
    *,
    checkpoint_sha256: str,
    image_count: int,
    data_root: Path,
    weights: Path,
    imgsz: int,
    device: str,
) -> bool:
    """Return true only when an existing diagnostic exactly matches this invocation."""
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
        settings = payload["settings"]
        provenance = payload["provenance"]
        return all(
            (
                provenance.get("weights_sha256") == checkpoint_sha256,
                int(settings.get("source_image_count")) == image_count,
                int(settings.get("evaluated_image_count")) == image_count,
                settings.get("is_subset_smoke_run") is False,
                settings.get("max_images") is None,
                int(settings.get("imgsz")) == imgsz,
                int(settings.get("batch")) == BATCH,
                int(settings.get("max_det")) == MAX_DET,
                str(settings.get("device")) == str(device),
                Path(str(settings.get("data_root"))).resolve() == data_root.resolve(),
                Path(str(settings.get("weights"))).resolve() == weights.resolve(),
                _same_number(settings.get("confidence_threshold"), CONFIDENCE),
                _same_number(settings.get("nms_iou_threshold"), NMS_IOU),
                _same_number(settings.get("matching_iou_threshold"), MATCH_IOU),
            )
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _run(script: str, *arguments: object) -> None:
    command = [sys.executable, "-u", str(ROOT / "scripts" / script)]
    command.extend(str(value) for value in arguments)
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="suite directory name under results/")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--raw-val-root",
        type=Path,
        default=ROOT / "data" / "raw" / "VisDrone2019-DET-val",
    )
    parser.add_argument(
        "--check-current-code",
        action="store_true",
        help="require the live training-code fingerprint to match protocol.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if Path(args.suite).name != args.suite or args.suite in {"", ".", ".."}:
        raise ValueError("--suite must be one directory name")
    suite_dir = ROOT / "results" / args.suite
    protocol_path = suite_dir / "protocol.json"
    protocol = _read_json(protocol_path)
    variants = [str(value) for value in protocol.get("variants") or []]
    seeds = [int(value) for value in protocol.get("seeds") or []]
    if not variants or not seeds or len(set(variants)) != len(variants) or len(set(seeds)) != len(seeds):
        raise ValueError("protocol must declare unique, non-empty variants and seeds")
    imgsz = int(protocol["imgsz"])
    val_count = int((protocol.get("actual_image_counts") or {})["val"])
    if imgsz < 1 or val_count < 1:
        raise ValueError("protocol imgsz and validation image count must be positive")
    raw_val_root = args.raw_val_root.resolve()
    if not (raw_val_root / "images").is_dir() or not (raw_val_root / "annotations").is_dir():
        raise FileNotFoundError(
            f"raw validation root must contain images/ and annotations/: {raw_val_root}"
        )

    verification, exit_code = verify_suite(
        args.suite,
        expected_runs=len(variants) * len(seeds),
        allow_incomplete=False,
        check_current_code=args.check_current_code,
    )
    if exit_code or verification["status"] != "complete":
        raise RuntimeError(
            "suite verification failed before postprocessing: "
            + json.dumps(
                {
                    "status": verification["status"],
                    "missing_runs": verification["missing_runs"],
                    "unexpected_runs": verification["unexpected_runs"],
                    "suite_checks": verification["suite_checks"],
                }
            )
        )
    _atomic_json(suite_dir / "verification.json", verification)

    records: dict[str, dict[str, Any]] = {}
    expected_ids = [f"{variant}_s{seed}" for seed in seeds for variant in variants]
    for run_id in expected_ids:
        record = _read_json(suite_dir / run_id / "metrics.json")
        if record.get("status") != "completed":
            raise RuntimeError(f"run is not completed: {run_id}")
        records[run_id] = record

    completed_diagnostics = 0
    for index, run_id in enumerate(expected_ids, start=1):
        record = records[run_id]
        weights = ROOT / "checkpoints" / args.suite / f"{run_id}.pt"
        output = suite_dir / run_id / "occlusion.json"
        if reusable_occlusion(
            output,
            checkpoint_sha256=str(record["checkpoint_sha256"]),
            image_count=val_count,
            data_root=raw_val_root,
            weights=weights,
            imgsz=imgsz,
            device=str(args.device),
        ):
            print(f"SKIP verified occlusion {run_id}", flush=True)
        else:
            print(f"OCCLUSION {index}/{len(expected_ids)} {run_id}", flush=True)
            _run(
                "evaluate_occlusion.py",
                "--weights", weights,
                "--data-root", raw_val_root,
                "--output", output,
                "--imgsz", imgsz,
                "--batch", BATCH,
                "--conf", CONFIDENCE,
                "--iou", NMS_IOU,
                "--match-iou", MATCH_IOU,
                "--max-det", MAX_DET,
                "--device", args.device,
            )
        if not reusable_occlusion(
            output,
            checkpoint_sha256=str(record["checkpoint_sha256"]),
            image_count=val_count,
            data_root=raw_val_root,
            weights=weights,
            imgsz=imgsz,
            device=str(args.device),
        ):
            raise RuntimeError(f"occlusion output failed postcondition checks: {run_id}")
        completed_diagnostics += 1

    visual_seed = min(seeds)
    visual_run_ids = [f"{variant}_s{visual_seed}" for variant in variants]
    visual_weights = [ROOT / "checkpoints" / args.suite / f"{run_id}.pt" for run_id in visual_run_ids]
    predictions = ROOT / "assets" / args.suite / "predictions.png"
    _run(
        "visualize_predictions.py",
        "--weights", *visual_weights,
        "--labels", *variants,
        "--output", predictions,
        "--data-root", raw_val_root,
        "--imgsz", imgsz,
        "--conf", CONFIDENCE,
        "--iou", NMS_IOU,
        "--max-det", MAX_DET,
        "--seed", 179,
        "--device", args.device,
    )
    selection = {
        "visualized_variants": variants,
        "visualized_seed": visual_seed,
        "image_selection_seed": 179,
        "selection_method": (
            f"All protocol variants at the preselected minimum protocol seed ({visual_seed}); "
            "qualitative images selected independently of predictions with fixed seed 179."
        ),
        "performance_aggregation": "All model performance aggregates use all protocol seeds.",
        "predictions_figure": predictions.relative_to(ROOT).as_posix(),
        "predictions_sidecar": predictions.with_suffix(".json").relative_to(ROOT).as_posix(),
        "occlusion_evaluated_run_count": completed_diagnostics,
        "expected_run_count": len(expected_ids),
    }
    _atomic_json(suite_dir / "selection.json", selection)
    _run("report.py", "--suite", args.suite)
    print("POSTPROCESS COMPLETE " + json.dumps(selection), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
