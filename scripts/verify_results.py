#!/usr/bin/env python3
"""CPU-only structural and provenance verification for an experiment suite."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))

import torch
from torch import nn

from visdrone_migration.model import AttentionStage, VARIANTS
from visdrone_migration.modules import GaborStem
from visdrone_migration.evidence import code_fingerprint
from visdrone_migration.spatial_blocks import haar_filters


LAYER_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
BANK_ATOL = 1e-3
CLASS_NAMES = (
    "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
    "awning-tricycle", "bus", "motor",
)
EFFICIENT_UNITS = {
    "pconv": "visdrone_migration.classic_blocks.FasterBlock",
    "star": "visdrone_migration.classic_blocks.StarBlock",
    "wtconv": "visdrone_migration.spatial_blocks.WTBlock",
    "lsconv": "visdrone_migration.spatial_blocks.LSConv",
}
EFFICIENT_DEPTHS = {4: 2, 6: 2, 8: 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualified_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def expected_layers(variant: str) -> tuple[dict[int, str], str | None]:
    factors = VARIANTS[variant]
    expected = {
        0: "ultralytics.nn.modules.conv.Conv",
        1: "ultralytics.nn.modules.conv.Conv",
        2: "ultralytics.nn.modules.block.C2f",
        3: "ultralytics.nn.modules.conv.Conv",
        4: "ultralytics.nn.modules.block.C2f",
        5: "ultralytics.nn.modules.conv.Conv",
        6: "ultralytics.nn.modules.block.C2f",
        7: "ultralytics.nn.modules.conv.Conv",
        8: "ultralytics.nn.modules.block.C2f",
        9: "ultralytics.nn.modules.block.SPPF",
    }
    bank_mode = None
    if "gabor" in factors or "random" in factors:
        expected[0] = "visdrone_migration.modules.GaborStem"
        bank_mode = "random" if "random" in factors else "gabor"
    if "c3k2" in factors:
        expected[2] = "visdrone_migration.modules.C3k2"
    if "se" in factors:
        expected[2] = "visdrone_migration.model.AttentionStage"
    if "ghost" in factors:
        for index in (4, 6, 8):
            expected[index] = "visdrone_migration.modules.C3Ghost"
    if "sppf" in factors:
        expected[9] = "visdrone_migration.modules.EnhancedSPPF"
    if variant == "ghostconv":
        for index in (1, 3, 5, 7):
            expected[index] = "visdrone_migration.modules.GhostConv"
    if variant in EFFICIENT_UNITS:
        for index in EFFICIENT_DEPTHS:
            expected[index] = "visdrone_migration.model.EfficientC2f"
    return expected, bank_mode


def verify_snapshot_hashes(suite_dir: Path, protocol: dict[str, object]) -> tuple[dict[str, bool], dict[str, object]]:
    snapshot = protocol.get("data_snapshot") or {}
    specifications = {
        "dataset_yaml": (snapshot.get("dataset_yaml"), protocol.get("data_yaml_sha256")),
        "preparation_manifest": (
            snapshot.get("preparation_manifest"), protocol.get("preparation_manifest_sha256")
        ),
    }
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    for name, (raw_path, expected_hash) in specifications.items():
        path = ROOT / str(raw_path) if raw_path else suite_dir / "data_snapshot" / (
            "dataset.yaml" if name == "dataset_yaml" else "preparation_manifest.json"
        )
        exists = path.is_file()
        actual_hash = sha256_file(path) if exists else None
        checks[f"snapshot_{name}_exists"] = exists
        checks[f"snapshot_{name}_sha256"] = exists and actual_hash == expected_hash
        details[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
        }
    return checks, details


def read_epoch_history(path: Path) -> tuple[list[int], list[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(float(row["epoch"])) for row in rows]
    ap = [float(row["metrics/mAP50-95(B)"]) for row in rows]
    if not all(math.isfinite(value) for value in ap):
        raise ValueError("CSV contains a non-finite AP50-95 value")
    return epochs, ap


def verify_filter_bank(
    saved_stem: GaborStem, mode: str, seed: int
) -> tuple[bool, dict[str, object]]:
    """Compare the saved fixed buffer with a freshly generated FP16 bank."""
    with torch.random.fork_rng(devices=[]):
        expected = GaborStem(
            saved_stem.c1,
            saved_stem.proj.conv.out_channels,
            mode=mode,
            seed=seed,
        ).filter_bank.half()
    actual = saved_stem.filter_bank.detach().cpu().half()
    max_abs_difference = float((actual.float() - expected.float()).abs().max())
    passed = torch.allclose(actual, expected, atol=BANK_ATOL, rtol=BANK_ATOL)
    return passed, {
        "mode": mode,
        "dtype_compared": "float16",
        "atol": BANK_ATOL,
        "rtol": BANK_ATOL,
        "max_abs_difference": max_abs_difference,
        "note": (
            "Tolerance permits harmless FP16/EMA buffer roundoff; a trained update "
            "large enough to change the fixed bank fails this check."
        ),
    }


def verify_run(
    suite: str,
    variant: str,
    seed: int,
    protocol: dict[str, object],
) -> dict[str, object]:
    run_id = f"{variant}_s{seed}"
    run_dir = ROOT / "results" / suite / run_id
    metrics_path = run_dir / "metrics.json"
    csv_path = run_dir / "epochs.csv"
    checkpoint_path = ROOT / "checkpoints" / suite / f"{run_id}.pt"
    checks: dict[str, bool] = {}
    details: dict[str, object] = {
        "metrics": metrics_path.relative_to(ROOT).as_posix(),
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "epochs_csv": csv_path.relative_to(ROOT).as_posix(),
    }
    errors: list[str] = []
    checkpoint: dict[str, Any] | None = None
    model = None
    try:
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        checks["record_status_completed"] = record.get("status") == "completed"
        checks["record_variant"] = record.get("variant") == variant
        checks["record_seed"] = record.get("seed") == seed
        checks["record_factors"] = tuple(record.get("factors", ())) == VARIANTS[variant]
        for key in ("epochs", "imgsz", "batch"):
            checks[f"protocol_{key}"] = record.get(key) == protocol.get(key)
        training = record.get("training") or {}
        checks["protocol_workers"] = training.get("workers") == protocol.get("workers")
        for key in (
            "data_fingerprint",
            "data_yaml_sha256",
            "training_code_sha256",
            "environment_fingerprint",
        ):
            checks[f"protocol_{key}"] = record.get(key) == protocol.get(key)

        checks["checkpoint_exists"] = checkpoint_path.is_file()
        checks["csv_exists"] = csv_path.is_file()
        if not checks["checkpoint_exists"] or not checks["csv_exists"]:
            raise FileNotFoundError("checkpoint or epochs CSV is missing")
        checkpoint_hash = sha256_file(checkpoint_path)
        details["checkpoint_sha256"] = checkpoint_hash
        checks["checkpoint_sha256"] = checkpoint_hash == record.get("checkpoint_sha256")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = checkpoint.get("ema") or checkpoint.get("model")
        if model is None:
            raise ValueError("checkpoint contains neither EMA nor model")
        checks["model_variant"] = getattr(model, "variant", None) == variant
        checks["yaml_variant"] = model.yaml.get("migration_variant") == variant
        checks["yaml_seed"] = model.yaml.get("migration_module_seed") == seed
        model_names = getattr(model, "names", {})
        if isinstance(model_names, dict):
            normalized_names = tuple(str(model_names.get(index)) for index in range(len(CLASS_NAMES)))
        else:
            normalized_names = tuple(str(value) for value in model_names)
        details["class_names"] = list(normalized_names)
        details["class_count"] = getattr(model, "nc", model.yaml.get("nc"))
        checks["class_names"] = normalized_names == CLASS_NAMES
        checks["class_count"] = getattr(model, "nc", model.yaml.get("nc")) == len(CLASS_NAMES)

        actual_layers = {index: qualified_name(model.model[index]) for index in LAYER_INDICES}
        expected, bank_mode = expected_layers(variant)
        details["actual_layers"] = {str(key): value for key, value in actual_layers.items()}
        details["expected_layers"] = {str(key): value for key, value in expected.items()}
        checks["active_layer_classes"] = actual_layers == expected
        factors = VARIANTS[variant]
        if "se" in factors:
            wrapper = model.model[2]
            expected_base = (
                "visdrone_migration.modules.C3k2"
                if "c3k2" in factors
                else "ultralytics.nn.modules.block.C2f"
            )
            checks["se_wrapper"] = (
                isinstance(wrapper, AttentionStage)
                and qualified_name(wrapper.base) == expected_base
                and qualified_name(wrapper.attention)
                == "visdrone_migration.modules.SEBlock"
            )
        else:
            checks["se_wrapper_absent"] = not isinstance(model.model[2], AttentionStage)

        if variant == "ghostconv":
            ghost_activations = {}
            for index in (1, 3, 5, 7):
                layer = model.model[index]
                active = (
                    qualified_name(layer) == "visdrone_migration.modules.GhostConv"
                    and isinstance(layer.cv1.act, nn.SiLU)
                    and isinstance(layer.cv2.act, nn.SiLU)
                )
                ghost_activations[str(index)] = active
            details["ghostconv_silu"] = ghost_activations
            checks["ghostconv_silu"] = all(ghost_activations.values())

        if variant in EFFICIENT_UNITS:
            expected_unit = EFFICIENT_UNITS[variant]
            stage_details = {}
            for index, expected_depth in EFFICIENT_DEPTHS.items():
                stage = model.model[index]
                units = list(getattr(stage, "m", ()))
                stage_details[str(index)] = {
                    "kind": getattr(stage, "kind", None),
                    "depth": len(units),
                    "unit_classes": [qualified_name(unit) for unit in units],
                }
            details["efficient_stages"] = stage_details
            checks["efficient_stage_kind"] = all(
                item["kind"] == variant for item in stage_details.values()
            )
            checks["efficient_stage_depth"] = all(
                stage_details[str(index)]["depth"] == depth
                for index, depth in EFFICIENT_DEPTHS.items()
            )
            checks["efficient_stage_units"] = all(
                all(name == expected_unit for name in stage_details[str(index)]["unit_classes"])
                for index in EFFICIENT_DEPTHS
            )
            if variant == "wtconv":
                maximum = 0.0
                buffers_pass = True
                buffer_count = 0
                for index in EFFICIENT_DEPTHS:
                    for unit in model.model[index].m:
                        spatial = unit.spatial
                        expected_bank = haar_filters(spatial.channels).to(
                            dtype=spatial.analysis_filter.dtype,
                            device=spatial.analysis_filter.device,
                        )
                        for name in ("analysis_filter", "synthesis_filter"):
                            actual = getattr(spatial, name)
                            difference = float((actual.float() - expected_bank.float()).abs().max())
                            maximum = max(maximum, difference)
                            buffers_pass = buffers_pass and torch.allclose(
                                actual, expected_bank, atol=BANK_ATOL, rtol=BANK_ATOL
                            )
                            buffer_count += 1
                checks["wtconv_fixed_haar_buffers"] = buffers_pass and buffer_count > 0
                details["wtconv_fixed_haar_buffers"] = {
                    "buffer_count": buffer_count,
                    "atol": BANK_ATOL,
                    "rtol": BANK_ATOL,
                    "max_abs_difference": maximum,
                }

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        details["parameter_count"] = parameter_count
        checks["parameter_count"] = parameter_count == record.get("parameters")

        if bank_mode is not None:
            stem = model.model[0]
            if isinstance(stem, GaborStem):
                passed, bank_details = verify_filter_bank(stem, bank_mode, seed)
                checks["fixed_filter_bank"] = passed
                details["fixed_filter_bank"] = bank_details
            else:
                checks["fixed_filter_bank"] = False
                details["fixed_filter_bank"] = {"error": "layer 0 is not GaborStem"}
        else:
            checks["fixed_filter_bank_absent"] = not isinstance(model.model[0], GaborStem)

        epochs, ap = read_epoch_history(csv_path)
        expected_epochs = int(record["epochs"])
        selected_epoch = int(record["selected_epoch"])
        checks["csv_exact_epochs"] = epochs == list(range(1, expected_epochs + 1))
        checks["selected_epoch_in_range"] = 1 <= selected_epoch <= expected_epochs
        if selected_epoch in epochs and ap:
            selected_ap = ap[epochs.index(selected_epoch)]
            maximum_ap = max(ap)
            difference = maximum_ap - selected_ap
            checks["selected_epoch_is_csv_maximum"] = difference <= 1e-5
            details["selection"] = {
                "selected_epoch": selected_epoch,
                "selected_csv_ap50_95": selected_ap,
                "maximum_csv_ap50_95": maximum_ap,
                "maximum_minus_selected": difference,
                "tolerance": 1e-5,
                "note": (
                    "The final EMA validation metric can differ from the rounded per-epoch CSV; "
                    "selection is checked within the CSV only."
                ),
            }
        else:
            checks["selected_epoch_is_csv_maximum"] = False
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        del model
        del checkpoint
        gc.collect()

    failed_checks = sorted(key for key, passed in checks.items() if not passed)
    return {
        "run_id": run_id,
        "variant": variant,
        "seed": seed,
        "passed": not errors and not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "errors": errors,
        "details": details,
    }


def verify_suite(
    suite: str,
    expected_runs: int | None = None,
    allow_incomplete: bool = False,
    check_current_code: bool = False,
) -> tuple[dict[str, object], int]:
    suite_dir = ROOT / "results" / suite
    protocol_path = suite_dir / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"suite protocol not found: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    variants = list(protocol.get("variants") or [])
    seeds = [int(seed) for seed in protocol.get("seeds") or []]
    expected_ids = [f"{variant}_s{seed}" for seed in seeds for variant in variants]
    protocol_run_count = len(expected_ids)
    if expected_runs is not None and protocol_run_count != expected_runs:
        raise ValueError(
            f"protocol defines {protocol_run_count} runs, but --expected-runs is {expected_runs}"
        )
    expected_runs = protocol_run_count
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if unknown_variants:
        raise ValueError(f"protocol contains unknown variant: {unknown_variants[0]}")

    present_ids = [
        run_id
        for run_id in expected_ids
        if (suite_dir / run_id / "metrics.json").is_file()
    ]
    missing_ids = [run_id for run_id in expected_ids if run_id not in present_ids]
    unexpected_ids = sorted(
        path.parent.name
        for path in suite_dir.glob("*/metrics.json")
        if path.parent.name not in set(expected_ids)
    )
    run_results = []
    for seed in seeds:
        for variant in variants:
            run_id = f"{variant}_s{seed}"
            if run_id in present_ids:
                run_results.append(verify_run(suite, variant, seed, protocol))

    snapshot_checks, snapshot_details = verify_snapshot_hashes(suite_dir, protocol)
    current_code_sha256 = code_fingerprint(ROOT)
    current_code_equals_protocol = current_code_sha256 == protocol.get("training_code_sha256")
    suite_checks = dict(snapshot_checks)
    if check_current_code:
        suite_checks["current_code_equals_protocol"] = current_code_equals_protocol
    all_present_pass = all(result["passed"] for result in run_results)
    suite_checks_pass = all(suite_checks.values())
    if unexpected_ids or not all_present_pass or not suite_checks_pass:
        status, exit_code = "failed", 1
    elif missing_ids:
        status, exit_code = ("partial", 0) if allow_incomplete else ("incomplete", 2)
    else:
        status, exit_code = "complete", 0
    verifier_path = Path(__file__).resolve()
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "suite": suite,
        "expected_run_count": expected_runs,
        "verified_run_count": len(run_results),
        "passed_run_count": sum(bool(result["passed"]) for result in run_results),
        "allow_incomplete": allow_incomplete,
        "check_current_code": check_current_code,
        "suite_checks": suite_checks,
        "missing_runs": missing_ids,
        "unexpected_runs": unexpected_ids,
        "runs": run_results,
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(protocol_path),
            "training_code_sha256": protocol.get("training_code_sha256"),
            "data_fingerprint": protocol.get("data_fingerprint"),
            "environment_fingerprint": protocol.get("environment_fingerprint"),
            "snapshot": snapshot_details,
            "current_code_sha256": current_code_sha256,
            "current_code_equals_protocol": current_code_equals_protocol,
        },
        "verifier_provenance": {
            "path": verifier_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(verifier_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "device_policy": "torch.load(map_location='cpu'); no model forward pass",
            "fixed_bank_tolerance": {"atol": BANK_ATOL, "rtol": BANK_ATOL},
        },
    }
    return payload, exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="pilot")
    parser.add_argument(
        "--expected-runs", type=int,
        help="optional assertion; by default the expected count is derived from protocol.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="verify present runs but label output partial when protocol runs are missing",
    )
    parser.add_argument(
        "--check-current-code",
        action="store_true",
        help="fail if the live training-code fingerprint differs from the archived protocol",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if Path(args.suite).name != args.suite or args.suite in {"", ".", ".."}:
        raise ValueError("--suite must be one directory name")
    if args.expected_runs is not None and args.expected_runs < 1:
        raise ValueError("--expected-runs must be positive")
    payload, exit_code = verify_suite(
        args.suite, args.expected_runs, args.allow_incomplete, args.check_current_code
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else ROOT / "results" / args.suite / "verification.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "verified": payload["verified_run_count"],
                "expected": payload["expected_run_count"],
                "output": str(output),
            }
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
