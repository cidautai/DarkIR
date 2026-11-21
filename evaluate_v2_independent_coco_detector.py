#!/usr/bin/env python3
# Team responsibility: Bhanu Prakash -- implementation/evaluation; Sanjay Naidu -- evaluator review.
"""Cross-detector validation of Balanced V2 DarkIR on official ExDark validation.

Uses an independent, frozen torchvision COCO-pretrained Faster R-CNN ResNet-50 FPN
(not the ExDark-trained V2 guidance teacher) to evaluate four inputs:
  1) original low-light image, 2) pretrained DarkIR, 3) V2-Control-1K,
  4) V2-Detection-1K.

Ground-truth parsing, ExDark split resolution, COCO category mapping and COCOeval
are imported from the audited evaluate_exdark_cocoeval.py evaluator. All four
conditions use exactly the same images, resizing, detector, score threshold and
COCOeval protocol. This script does not modify repository files or checkpoints.

Run from the DarkIR repository root.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)


def load_base_module():
    repo = Path.cwd().resolve()
    if not (repo / "evaluate_exdark_cocoeval.py").is_file():
        raise RuntimeError(
            "Run this script from the DarkIR repository root (the directory containing "
            "evaluate_exdark_cocoeval.py)."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import evaluate_exdark_cocoeval as base
    return base


def load_darkir(checkpoint_path: Path, device: torch.device, base):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DarkIR checkpoint not found: {checkpoint_path}")
    model = base.DarkIR(
        img_channel=3, width=32, middle_blk_num_enc=2, middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9], extra_depth_wise=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint format: {checkpoint_path}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def restore(model: nn.Module, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run DarkIR with right/bottom replicate padding to a multiple of eight."""
    x = image.unsqueeze(0).to(device, non_blocking=True)
    height, width = x.shape[-2:]
    pad_h, pad_w = (-height) % 8, (-width) % 8
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    out = model(x).clamp(0.0, 1.0)[..., :height, :width]
    return out.squeeze(0)


def build_exdark_validation(args, base):
    split_code = 2  # Official validation split only; do not use test for tuning.
    records = base.parse_split_file(args.split_file, split_code)
    if len(records) != 1800:
        raise RuntimeError(
            f"Official ExDark validation should contain 1800 records, found {len(records)}."
        )

    image_index = base.build_image_index(args.images_root)
    annotation_index = base.build_annotation_index(args.annotations_root)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    category_names = weights.meta["categories"]
    exdark_to_coco_id = {}
    for name in base.EXDARK_CLASSES:
        coco_name = base.EXDARK_TO_COCO_NAME[name.lower()]
        if coco_name not in category_names:
            raise RuntimeError(f"COCO detector weights do not contain category {coco_name!r}")
        exdark_to_coco_id[name.lower()] = category_names.index(coco_name)

    prepared, gt_by_image = [], []
    counts = defaultdict(int)
    total_boxes = 0
    empty_images = []
    for rel, class_name in records:
        image_path = base.resolve_image(rel, class_name, args.images_root, image_index)
        annotation_path = base.resolve_annotation(
            image_path, class_name, args.annotations_root, annotation_index
        )
        with Image.open(image_path) as image:
            orig_w, orig_h = image.size
            image.verify()
        if args.max_side > 0 and max(orig_w, orig_h) > args.max_side:
            scale = args.max_side / max(orig_w, orig_h)
            new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
        else:
            new_w, new_h = orig_w, orig_h
        gt = base.parse_gt_file(
            annotation_path, new_w, new_h, exdark_to_coco_id,
            scale_x=new_w / orig_w, scale_y=new_h / orig_h,
        )
        if not gt:
            empty_images.append((rel, annotation_path))
        for category_id, boxes in gt.items():
            counts[category_id] += len(boxes)
            total_boxes += len(boxes)
        prepared.append((image_path, new_w, new_h))
        gt_by_image.append(gt)

    if empty_images:
        examples = "\n".join(f"  {rel} -> {path}" for rel, path in empty_images[:10])
        raise RuntimeError(
            f"Found {len(empty_images)} validation images without parsed GT boxes; stopping:\n{examples}"
        )
    if total_boxes != 6212:
        raise RuntimeError(
            f"Expected 6212 GT boxes in official ExDark validation, parsed {total_boxes}."
        )

    print(
        f"Preflight passed: official ExDark validation, {len(prepared)} images, "
        f"{total_boxes} boxes, {len(counts)} classes.", flush=True,
    )
    for class_name in base.EXDARK_CLASSES:
        cid = exdark_to_coco_id[class_name.lower()]
        print(f"  {class_name:12s}: {counts.get(cid, 0)} boxes")
    return records, prepared, gt_by_image, exdark_to_coco_id, weights


def read_prediction_cache(path: Path, prepared, num_images: int):
    cache = json.loads(path.read_text(encoding="utf-8"))
    required = {"original_low_light", "darkir_pretrained", "v2_control_1k", "v2_detection_1k"}
    if cache.get("split") != "val" or cache.get("num_images") != num_images:
        raise RuntimeError("Prediction cache is not for the current official ExDark validation split.")
    if set(cache.get("predictions", {})) != required:
        raise RuntimeError("Prediction cache must contain all four independent-detector conditions.")
    saved = cache.get("prepared", [])
    if len(saved) != len(prepared):
        raise RuntimeError("Prediction-cache image count differs from the current validation split.")
    for i, ((path_now, width_now, height_now), old) in enumerate(zip(prepared, saved)):
        if (old.get("path") != str(path_now) or int(old.get("width", -1)) != width_now
                or int(old.get("height", -1)) != height_now):
            raise RuntimeError(f"Prediction-cache metadata mismatch at image {i}.")
    return cache["predictions"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, default=Path("data/datasets/ExDark/images"))
    parser.add_argument("--annotations-root", type=Path, default=Path("data/datasets/ExDark/Groundtruth"))
    parser.add_argument("--split-file", type=Path, default=Path("data/datasets/ExDark/Groundtruth/imageclasslist.txt"))
    parser.add_argument("--pretrained", type=Path, default=Path("models/DarkIR_384.pt"))
    parser.add_argument("--control-checkpoint", type=Path, default=Path("models/balanced_v2_1k/DarkIR_v2_control_pilot_1000.pt"))
    parser.add_argument("--detection-checkpoint", type=Path, default=Path("models/balanced_v2_1k/DarkIR_v2_detection_pilot_1000.pt"))
    parser.add_argument("--max-side", type=int, default=640,
                        help="640 matches the Balanced V2 ExDark guidance/evaluation resolution; 0 keeps native size. Use 1280 only for legacy V1-resolution comparison.")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v2_independent_coco_detector"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--reuse-predictions-cache", type=Path, default=None)
    args = parser.parse_args()

    if args.max_side < 0 or not (0.0 <= args.score_threshold <= 1.0):
        raise ValueError("--max-side must be nonnegative and score threshold must be in [0,1].")

    base = load_base_module()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records, prepared, gt_by_image, exdark_to_coco_id, weights = build_exdark_validation(args, base)
    if args.preflight_only:
        print("PREFLIGHT ONLY PASSED. No models loaded and no inference run.")
        return
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available; use the GPU environment for evaluation.")
    try:
        from pycocotools.coco import COCO as _COCO  # noqa: F401
        from pycocotools.cocoeval import COCOeval as _COCOeval  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install pycocotools in the DarkIR environment: pip install pycocotools") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "exdark_val_independent_predictions.json"
    if args.reuse_predictions_cache:
        predictions = read_prediction_cache(args.reuse_predictions_cache, prepared, len(records))
        print(f"Reusing prediction cache: {args.reuse_predictions_cache}", flush=True)
    else:
        print(f"Independent detector: torchvision Faster R-CNN ResNet-50 FPN, COCO weights.", flush=True)
        print(f"Device: {torch.cuda.get_device_name(0)} | max_side={args.max_side} | score_threshold={args.score_threshold}", flush=True)
        detector = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None).to(device).eval()
        detector.roi_heads.score_thresh = args.score_threshold
        detector.roi_heads.detections_per_img = 100
        for parameter in detector.parameters():
            parameter.requires_grad_(False)

        models = {
            "darkir_pretrained": load_darkir(args.pretrained, device, base),
            "v2_control_1k": load_darkir(args.control_checkpoint, device, base),
            "v2_detection_1k": load_darkir(args.detection_checkpoint, device, base),
        }
        print("Loaded all three DarkIR checkpoints with strict=True.", flush=True)
        conditions = ["original_low_light", *models.keys()]
        predictions = {name: [] for name in conditions}
        mapped_ids = set(exdark_to_coco_id.values())
        started = time.time()

        for image_id, (image_path, width, height) in enumerate(prepared):
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.BILINEAR)
                tensor = to_tensor(image)
            tensor_gpu = tensor.to(device, non_blocking=True)
            with torch.inference_mode():
                condition_images = {"original_low_light": tensor_gpu}
                for name, model in models.items():
                    condition_images[name] = restore(model, tensor, device)
                for name in conditions:
                    output = detector([condition_images[name]])[0]
                    boxes = output["boxes"].detach().cpu().numpy()
                    labels = output["labels"].detach().cpu().numpy()
                    scores = output["scores"].detach().cpu().numpy()
                    for box, label, score in zip(boxes, labels, scores):
                        label = int(label)
                        score = float(score)
                        if label not in mapped_ids or score < args.score_threshold:
                            continue
                        predictions[name].append({
                            "image_id": image_id,
                            "category_id": label,
                            "score": score,
                            "box": [float(value) for value in box],
                        })
            if (image_id + 1) % 50 == 0 or image_id + 1 == len(prepared):
                elapsed = time.time() - started
                print(f"processed {image_id+1}/{len(prepared)} | {elapsed/60:.1f} min", flush=True)
            del tensor_gpu

        cache = {
            "dataset": "ExDark", "split": "val", "num_images": len(records),
            "max_side": args.max_side, "score_threshold": args.score_threshold,
            "detector": "torchvision Faster R-CNN ResNet-50 FPN, COCO-pretrained, frozen",
            "prepared": [
                {"path": str(path), "width": int(width), "height": int(height)}
                for path, width, height in prepared
            ],
            "predictions": predictions,
        }
        cache_path.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
        print(f"Saved raw predictions cache: {cache_path}", flush=True)
        del detector, models
        torch.cuda.empty_cache()

    summaries, _, _ = base.evaluate_with_cocoeval(
        prepared, gt_by_image, predictions, exdark_to_coco_id
    )
    output = {
        "dataset": "ExDark", "split": "official_validation", "num_images": len(records),
        "num_ground_truth_boxes": sum(sum(len(v) for v in x.values()) for x in gt_by_image),
        "detector": "torchvision Faster R-CNN ResNet-50 FPN, COCO-pretrained and frozen (independent of ExDark guidance teacher)",
        "max_side": args.max_side,
        "score_threshold": args.score_threshold,
        "class_mapping": {k: base.EXDARK_TO_COCO_NAME[k] for k in base.EXDARK_TO_COCO_NAME},
        "checkpoints": {
            "darkir_pretrained": str(args.pretrained),
            "v2_control_1k": str(args.control_checkpoint),
            "v2_detection_1k": str(args.detection_checkpoint),
        },
        "metric_protocol": "pycocotools COCOeval bbox; IoU=.50:.05:.95; 101 recall thresholds; maxDets=[1,10,100]",
        "results": summaries,
        "predictions_cache": str(args.reuse_predictions_cache or cache_path),
    }
    output_path = args.output_dir / "exdark_val_independent_coco_metrics.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("\nIndependent COCO-detector results:")
    print(f"{'Condition':22s} {'mAP50':>8s} {'mAP50:95':>10s} {'mAP75':>8s}")
    for name, values in summaries.items():
        print(f"{name:22s} {values['mAP50']:8.4f} {values['mAP50_95']:10.4f} {values['mAP75']:8.4f}")
    print(f"\nSaved metrics: {output_path}", flush=True)
    print("NOTE: this is an independent COCO-pretrained detector check; compare conditions only within this fixed protocol.")


if __name__ == "__main__":
    main()
