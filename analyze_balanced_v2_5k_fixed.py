#!/usr/bin/env python3
# Team responsibility: Bhanu Prakash -- per-class result reconciliation.
"""Compare per-class Balanced V2 5K results from both ExDark evaluators.

Reads the two existing JSON metric files only. The ExDark-teacher evaluator
schema does not store DarkIR checkpoint paths; the independent evaluator does.
The latter is used to verify that the compared DarkIR checkpoints are the 5K
artifacts. No GPU inference, training, or checkpoint modification is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

EXPECTED_CLASSES = {
    "bicycle", "boat", "bottle", "bus", "car", "cat",
    "chair", "cup", "dog", "motorbike", "people", "table",
}


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} JSON not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{label} JSON root must be an object")
    return data


def validate_dataset_metadata(data: dict[str, Any], label: str) -> None:
    if data.get("dataset") != "ExDark":
        raise ValueError(f"{label}: expected dataset='ExDark', got {data.get('dataset')!r}")
    if data.get("split") != "official_validation":
        raise ValueError(f"{label}: expected split='official_validation', got {data.get('split')!r}")
    if data.get("num_images") != 1800:
        raise ValueError(f"{label}: expected 1800 validation images, got {data.get('num_images')!r}")
    if data.get("num_ground_truth_boxes") != 6212:
        raise ValueError(
            f"{label}: expected 6212 ground-truth boxes, got {data.get('num_ground_truth_boxes')!r}"
        )


def verify_independent_checkpoint_paths(data: dict[str, Any]) -> None:
    paths = data.get("checkpoints")
    if not isinstance(paths, dict):
        raise KeyError("Independent-detector JSON is missing its 'checkpoints' object")
    # The evaluator currently keeps historical '_1k' condition/checkpoint keys,
    # even when the paths passed on the command line refer to the 5K files.
    required = ("v2_control_1k", "v2_detection_1k")
    for key in required:
        value = paths.get(key)
        if not isinstance(value, str) or not value:
            raise KeyError(f"Independent-detector JSON is missing checkpoints[{key!r}]")
        path = Path(value)
        if "balanced_v2_5k" not in value or "5000" not in path.name:
            raise ValueError(f"{key} does not point to a 5K checkpoint: {value}")
    if data.get("max_side") != 640:
        raise ValueError(f"Independent-detector max_side must be 640, got {data.get('max_side')!r}")
    detector = str(data.get("detector", "")).lower()
    if "coco-pretrained" not in detector or "independent" not in detector:
        raise ValueError("Independent-detector metadata does not identify the COCO-pretrained independent detector")


def per_class_map(results: dict[str, Any], condition: str, evaluator_label: str) -> dict[str, dict[str, float]]:
    if condition not in results:
        raise KeyError(
            f"{evaluator_label}: missing condition {condition!r}; available: {list(results)}"
        )
    per_class = results[condition].get("per_class")
    if not isinstance(per_class, dict):
        raise KeyError(f"{evaluator_label}: no per_class map for condition {condition!r}")

    normalized: dict[str, dict[str, float]] = {}
    for raw_name, values in per_class.items():
        if not isinstance(values, dict):
            continue
        name = str(raw_name).strip().lower().replace(" ", "")
        if name == "motorcycle":
            name = "motorbike"
        if name == "person":
            name = "people"
        if name == "diningtable":
            name = "table"
        if name not in EXPECTED_CLASSES:
            continue
        row: dict[str, float] = {}
        for metric in ("AP50", "AP50_95"):
            value = values.get(metric)
            if value is not None:
                row[metric] = float(value)
        normalized[name] = row

    if set(normalized) != EXPECTED_CLASSES:
        missing = sorted(EXPECTED_CLASSES - set(normalized))
        extra = sorted(set(normalized) - EXPECTED_CLASSES)
        raise ValueError(
            f"{evaluator_label}: expected exactly 12 ExDark classes; missing={missing}, extra={extra}"
        )
    for class_name, metrics in normalized.items():
        if "AP50_95" not in metrics:
            raise KeyError(f"{evaluator_label}: AP50_95 missing for {class_name}")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-json", type=Path,
        default=Path("results/balanced_v2_5k_validation/exdark_official_validation.json"),
    )
    parser.add_argument(
        "--independent-json", type=Path,
        default=Path("results/v2_independent_coco_detector_5k/exdark_val_independent_coco_metrics.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/balanced_v2_5k_validation/5k_per_class_comparison.csv"),
    )
    args = parser.parse_args()

    teacher = load_json(args.teacher_json, "ExDark-teacher evaluation")
    independent = load_json(args.independent_json, "Independent COCO evaluation")

    validate_dataset_metadata(teacher, "ExDark-teacher evaluation")
    validate_dataset_metadata(independent, "Independent COCO evaluation")
    verify_independent_checkpoint_paths(independent)

    # Important schema difference: evaluate_balanced_v2.py writes results and
    # dataset metadata, but intentionally does NOT write restoration checkpoint
    # paths. The user's saved terminal log records those 5K inputs. The
    # independent evaluator persists them and they are verified above.
    teacher_results = teacher.get("results")
    independent_results = independent.get("results")
    if not isinstance(teacher_results, dict) or not isinstance(independent_results, dict):
        raise KeyError("One of the JSON files is missing its top-level 'results' object")

    t_control = per_class_map(teacher_results, "v2_control", "Teacher")
    t_detection = per_class_map(teacher_results, "v2_detection", "Teacher")
    i_control = per_class_map(independent_results, "v2_control_1k", "Independent COCO")
    i_detection = per_class_map(independent_results, "v2_detection_1k", "Independent COCO")

    rows: list[dict[str, Any]] = []
    for cls in sorted(EXPECTED_CLASSES):
        rows.append({
            "class": cls,
            "teacher_control_AP50": t_control[cls].get("AP50", ""),
            "teacher_detection_AP50": t_detection[cls].get("AP50", ""),
            "teacher_delta_AP50": (
                t_detection[cls]["AP50"] - t_control[cls]["AP50"]
                if "AP50" in t_detection[cls] and "AP50" in t_control[cls] else ""
            ),
            "teacher_control_AP50_95": t_control[cls]["AP50_95"],
            "teacher_detection_AP50_95": t_detection[cls]["AP50_95"],
            "teacher_delta_AP50_95": t_detection[cls]["AP50_95"] - t_control[cls]["AP50_95"],
            "independent_control_AP50": i_control[cls].get("AP50", ""),
            "independent_detection_AP50": i_detection[cls].get("AP50", ""),
            "independent_delta_AP50": (
                i_detection[cls]["AP50"] - i_control[cls]["AP50"]
                if "AP50" in i_detection[cls] and "AP50" in i_control[cls] else ""
            ),
            "independent_control_AP50_95": i_control[cls]["AP50_95"],
            "independent_detection_AP50_95": i_detection[cls]["AP50_95"],
            "independent_delta_AP50_95": i_detection[cls]["AP50_95"] - i_control[cls]["AP50_95"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Validation metadata: ExDark official validation, 1,800 images, 6,212 boxes.")
    print("Independent checkpoints verified:")
    print(f"  Control:   {independent['checkpoints']['v2_control_1k']}")
    print(f"  Detection: {independent['checkpoints']['v2_detection_1k']}")
    print("Teacher-evaluation JSON has no checkpoint-path fields; 5K paths were recorded in the supplied terminal log.")
    print("Independent JSON condition names retain the historical '_1k' suffix; actual 5K paths were verified.")
    print()
    print(f"{'Class':12} {'Teacher Δ AP50:95':>19} {'Independent Δ AP50:95':>24}")
    print("-" * 60)
    for row in rows:
        print(
            f"{row['class']:12} {row['teacher_delta_AP50_95']:+19.4f} "
            f"{row['independent_delta_AP50_95']:+24.4f}"
        )

    for label, key in (
        ("Teacher", "teacher_delta_AP50_95"),
        ("Independent", "independent_delta_AP50_95"),
    ):
        deltas = [float(row[key]) for row in rows]
        improved = sum(value > 0 for value in deltas)
        worse = sum(value < 0 for value in deltas)
        unchanged = sum(value == 0 for value in deltas)
        print(
            f"\n{label}: improved={improved}/12, worse={worse}/12, "
            f"unchanged={unchanged}/12, mean per-class AP50:95 delta={mean(deltas):+.4f}"
        )

    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
