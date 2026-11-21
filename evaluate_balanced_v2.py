#!/usr/bin/env python3
# Team responsibility: Bhanu Prakash -- implementation/evaluation; Sanjay Naidu -- evaluator review.
"""Validate Balanced V2 DarkIR pilots on reserved LOLBlur and ExDark splits.

Run from the DarkIR repository root. This script is read-only with respect to
training data and checkpoints; it writes metrics/prediction artifacts only under
--output-dir.

LOLBlur: full-resolution paired inference; reports per-image macro PSNR, SSIM,
and AlexNet LPIPS for low-light input, V2-Control, and V2-Detection.
ExDark: official validation manifest; evaluates low-light input and both V2
outputs using the same frozen ExDark-trained 12-class Faster R-CNN teacher used
for V2 detection guidance. Reports pycocotools COCO bbox AP.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def pil_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, int, int]:
    """Replicate-pad BCHW input for DarkIR and return its original H/W."""
    height, width = x.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, height, width


def load_darkir(checkpoint_path: Path, device: torch.device) -> nn.Module:
    repo_root = Path.cwd().resolve()
    if not (repo_root / "archs" / "DarkIR.py").is_file():
        raise RuntimeError("Run this script from the DarkIR repository root")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from archs.DarkIR import DarkIR

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DarkIR checkpoint not found: {checkpoint_path}")
    model = DarkIR(
        img_channel=3, width=32,
        middle_blk_num_enc=2, middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9], extra_depth_wise=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def restore(model: nn.Module, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = image.unsqueeze(0).to(device, non_blocking=True)
    x_pad, height, width = pad_to_multiple(x)
    with torch.inference_mode():
        out = model(x_pad).clamp(0.0, 1.0)[..., :height, :width]
    return out.squeeze(0)


def make_gaussian_window(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(11, device=device, dtype=dtype) - 5
    one_d = torch.exp(-(coords * coords) / (2.0 * 1.5 * 1.5))
    one_d = one_d / one_d.sum()
    window = torch.outer(one_d, one_d)
    return window.view(1, 1, 11, 11).expand(channels, 1, 11, 11).contiguous()


def ssim_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """RGB SSIM with an 11x11, sigma=1.5 Gaussian window and data_range=1."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("SSIM expects equal CHW tensors")
    x = prediction.unsqueeze(0).float()
    y = target.unsqueeze(0).float()
    channels = x.shape[1]
    if min(x.shape[-2:]) < 11:
        raise ValueError(f"Image too small for SSIM 11x11 window: {tuple(x.shape[-2:])}")
    window = make_gaussian_window(channels, x.device, x.dtype)
    # Reflect padding matches the common Gaussian-window SSIM boundary convention.
    x = F.pad(x, (5, 5, 5, 5), mode="reflect")
    y = F.pad(y, (5, 5, 5, 5), mode="reflect")
    mu_x = F.conv2d(x, window, groups=channels)
    mu_y = F.conv2d(y, window, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, groups=channels) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return float(score.mean().detach().cpu())


def psnr_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(F.mse_loss(prediction.float(), target.float()).detach().cpu())
    if mse <= 1e-12:
        return 120.0
    return 10.0 * math.log10(1.0 / mse)


def validate_lolblur(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS is required. Install it in the DarkIR environment: pip install lpips==0.1.4") from exc

    manifest = read_json(args.lolblur_manifest)
    if manifest.get("dataset") != "LOLBlur" or manifest.get("format_version") != 1:
        raise ValueError("Unsupported LOLBlur validation manifest")
    if manifest.get("split") != "internal_validation":
        raise ValueError("Expected the scene-disjoint lolblur_internal_val.json manifest")
    pairs = manifest.get("pairs", [])
    if len(pairs) != int(manifest.get("num_pairs", -1)) or not pairs:
        raise ValueError("LOLBlur validation manifest pair count is invalid")
    scenes = {item.get("scene") for item in pairs}
    if len(scenes) != int(manifest.get("num_scenes", -1)):
        raise ValueError("LOLBlur validation manifest scene count is invalid")

    control = load_darkir(args.control_checkpoint, device)
    detection = load_darkir(args.detection_checkpoint, device)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)

    names = ["original_low_light", "v2_control", "v2_detection"]
    totals = {name: {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0} for name in names}
    started = time.time()
    root = args.lolblur_root

    for index, record in enumerate(pairs, start=1):
        low_path = root / record["low"]
        ref_path = root / record["reference"]
        if not low_path.is_file() or not ref_path.is_file():
            raise FileNotFoundError(f"Missing LOLBlur pair: {low_path} | {ref_path}")
        with Image.open(low_path) as im:
            low = pil_tensor(im)
        with Image.open(ref_path) as im:
            target = pil_tensor(im)
        if low.shape != target.shape:
            raise RuntimeError(f"LOLBlur pair dimensions differ: {low_path} vs {ref_path}")
        if min(low.shape[-2:]) < 64:
            raise RuntimeError(f"LPIPS AlexNet requires larger validation images; got {tuple(low.shape[-2:])} at {low_path}")

        restored = {
            "original_low_light": low.to(device),
            "v2_control": restore(control, low, device),
            "v2_detection": restore(detection, low, device),
        }
        target_gpu = target.to(device)
        for name, prediction in restored.items():
            prediction = prediction.clamp(0.0, 1.0)
            with torch.inference_mode():
                lp = perceptual(
                    prediction.unsqueeze(0) * 2.0 - 1.0,
                    target_gpu.unsqueeze(0) * 2.0 - 1.0,
                ).mean()
            totals[name]["psnr"] += psnr_value(prediction, target_gpu)
            totals[name]["ssim"] += ssim_value(prediction, target_gpu)
            totals[name]["lpips"] += float(lp.detach().cpu())
        if index % 100 == 0 or index == len(pairs):
            elapsed = time.time() - started
            print(f"LOLBlur validation {index}/{len(pairs)} | {elapsed/60:.1f} min", flush=True)
        del restored, target_gpu

    metrics = {
        "dataset": "LOLBlur",
        "split": "internal_validation",
        "num_pairs": len(pairs),
        "num_scenes": len(scenes),
        "aggregation": "macro mean of per-image metrics",
        "ssim_protocol": "RGB Gaussian-window SSIM, 11x11 window, sigma=1.5, data_range=1",
        "lpips_protocol": "LPIPS AlexNet, RGB inputs normalized to [-1,1]",
        "results": {
            name: {metric: value / len(pairs) for metric, value in vals.items()}
            for name, vals in totals.items()
        },
        "checkpoints": {
            "v2_control": str(args.control_checkpoint),
            "v2_detection": str(args.detection_checkpoint),
        },
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    output = args.output_dir / "lolblur_internal_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(args.output_dir / "balanced_v2_summary.csv", "LOLBlur", metrics["results"])
    print("\nLOLBlur internal validation:")
    for name, values in metrics["results"].items():
        print(f"  {name:18s} PSNR={values['psnr']:.4f} SSIM={values['ssim']:.5f} LPIPS={values['lpips']:.5f}")
    print(f"Saved: {output}", flush=True)

    del control, detection, perceptual
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def load_exdark_teacher(checkpoint_path: Path, device: torch.device, score_threshold: float) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ExDark teacher checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("class_names") != EXDARK_CLASSES:
        raise ValueError("Teacher class ordering differs from the Balanced V2 12-class mapping")
    if checkpoint.get("num_classes_with_background") != 13:
        raise ValueError("Expected an ExDark teacher with 13 outputs (background + 12 classes)")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Teacher checkpoint is missing model_state_dict")
    teacher = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = teacher.roi_heads.box_predictor.cls_score.in_features
    teacher.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=13)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.roi_heads.score_thresh = score_threshold
    teacher.roi_heads.detections_per_img = 100
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def coco_metrics(
    gt_by_image: list[dict[int, np.ndarray]],
    predictions: dict[str, list[dict[str, Any]]],
    image_sizes: list[tuple[int, int]],
) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("pycocotools is required. Install it in the DarkIR environment: pip install pycocotools") from exc

    categories = [{"id": i + 1, "name": name.lower()} for i, name in enumerate(EXDARK_CLASSES)]
    if len(image_sizes) != len(gt_by_image):
        raise ValueError("COCO image-size metadata count does not match GT count")
    images = [
        {"id": i, "file_name": str(i), "width": int(image_sizes[i][0]), "height": int(image_sizes[i][1])}
        for i in range(len(gt_by_image))
    ]
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for image_id, gt in enumerate(gt_by_image):
        for category_id, boxes in gt.items():
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box]
                width, height = x2 - x1, y2 - y1
                if width <= 0 or height <= 0:
                    continue
                annotations.append({
                    "id": ann_id, "image_id": image_id, "category_id": int(category_id),
                    "bbox": [x1, y1, width, height], "area": width * height, "iscrowd": 0,
                })
                ann_id += 1
    coco_gt = COCO()
    coco_gt.dataset = {"info": {"description": "ExDark official validation"}, "images": images,
                       "annotations": annotations, "categories": categories}
    coco_gt.createIndex()

    summaries: dict[str, Any] = {}
    cat_ids = [item["id"] for item in categories]
    for condition, preds in predictions.items():
        rows = []
        for pred in preds:
            x1, y1, x2, y2 = [float(v) for v in pred["box"]]
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            rows.append({
                "image_id": int(pred["image_id"]), "category_id": int(pred["category_id"]),
                "bbox": [x1, y1, width, height], "score": float(pred["score"]),
            })
        if rows:
            coco_dt = coco_gt.loadRes(rows)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {"images": images, "categories": categories, "annotations": []}
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = [image["id"] for image in images]
        evaluator.params.catIds = cat_ids
        evaluator.params.iouThrs = np.linspace(0.50, 0.95, 10)
        evaluator.params.recThrs = np.linspace(0.0, 1.0, 101)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        precision = evaluator.eval["precision"]  # T x R x K x A x M
        per_class: dict[str, Any] = {}
        for class_index, name in enumerate(EXDARK_CLASSES):
            p = precision[:, :, class_index, 0, 2]
            valid = p[p > -1]
            p50 = precision[0, :, class_index, 0, 2]
            p75 = precision[5, :, class_index, 0, 2]
            valid50, valid75 = p50[p50 > -1], p75[p75 > -1]
            per_class[name] = {
                "AP50": float(valid50.mean()) if valid50.size else None,
                "AP75": float(valid75.mean()) if valid75.size else None,
                "AP50_95": float(valid.mean()) if valid.size else None,
            }
        summaries[condition] = {
            "mAP50_95": float(evaluator.stats[0]),
            "mAP50": float(evaluator.stats[1]),
            "mAP75": float(evaluator.stats[2]),
            "per_class": per_class,
        }
    return summaries


def validate_exdark(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO as _COCO_CHECK  # noqa: F401
        from pycocotools.cocoeval import COCOeval as _COCOEVAL_CHECK  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pycocotools is required. Install it: pip install pycocotools") from exc

    manifest = read_json(args.exdark_manifest)
    if manifest.get("dataset") != "ExDark" or manifest.get("format_version") != 1:
        raise ValueError("Unsupported ExDark validation manifest")
    if manifest.get("split") != "official_validation":
        raise ValueError("Expected exdark_official_val.json, not teacher or detection-supervision data")
    if manifest.get("classes") != EXDARK_CLASSES or manifest.get("label_ids") != list(range(1, 13)):
        raise ValueError("ExDark manifest class names/IDs differ from the V2 teacher")
    records = manifest.get("records", [])
    expected_images = int(manifest.get("num_images", -1))
    expected_boxes = int(manifest.get("num_boxes", -1))
    if len(records) != expected_images or not records:
        raise ValueError("ExDark official validation manifest image count is invalid")
    actual_boxes = sum(len(item.get("boxes", [])) for item in records)
    if actual_boxes != expected_boxes:
        raise ValueError(f"ExDark manifest box count mismatch: expected {expected_boxes}, found {actual_boxes}")
    if expected_images != 1800 or expected_boxes != 6212:
        raise ValueError(
            f"Unexpected official ExDark validation manifest size: {expected_images} images / {expected_boxes} boxes; "
            "expected the audited 1800 / 6212 split"
        )

    control = load_darkir(args.control_checkpoint, device)
    detection = load_darkir(args.detection_checkpoint, device)
    teacher = load_exdark_teacher(args.teacher_checkpoint, device, args.score_threshold)
    predictions: dict[str, list[dict[str, Any]]] = {
        "original_low_light": [], "v2_control": [], "v2_detection": []
    }
    gt_by_image: list[dict[int, np.ndarray]] = []
    image_sizes: list[tuple[int, int]] = []
    started = time.time()

    for image_id, record in enumerate(records):
        image_path = args.exdark_root / record["image"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            orig_w, orig_h = image.size
            if int(record["width"]) != orig_w or int(record["height"]) != orig_h:
                raise RuntimeError(f"Image dimensions differ from manifest for {image_path}")
            max_side = args.exdark_max_side
            if max_side > 0 and max(orig_w, orig_h) > max_side:
                scale = max_side / max(orig_w, orig_h)
                new_w = max(1, round(orig_w * scale))
                new_h = max(1, round(orig_h * scale))
                image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            else:
                new_w, new_h = orig_w, orig_h
        image_sizes.append((new_w, new_h))
        sx, sy = new_w / orig_w, new_h / orig_h
        gt: dict[int, list[list[float]]] = {}
        for row in record["boxes"]:
            x1, y1, x2, y2, class_id = row
            class_id = int(class_id)
            if not 1 <= class_id <= 12:
                raise RuntimeError(f"Invalid class ID {class_id} in {image_path}")
            gt.setdefault(class_id, []).append([float(x1) * sx, float(y1) * sy, float(x2) * sx, float(y2) * sy])
        gt_by_image.append({key: np.asarray(value, dtype=np.float32) for key, value in gt.items()})

        image_tensor = pil_tensor(image)
        # Run DarkIR outputs one at a time to keep GPU memory bounded.
        with torch.inference_mode():
            raw = image_tensor.to(device, non_blocking=True)
            pred_raw = teacher([raw])[0]
            restored_control = restore(control, image_tensor, device)
            pred_control = teacher([restored_control])[0]
            restored_detection = restore(detection, image_tensor, device)
            pred_detection = teacher([restored_detection])[0]

        for condition, pred in (
            ("original_low_light", pred_raw),
            ("v2_control", pred_control),
            ("v2_detection", pred_detection),
        ):
            boxes = pred["boxes"].detach().cpu().numpy()
            labels = pred["labels"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()
            for box, label, score in zip(boxes, labels, scores):
                label, score = int(label), float(score)
                if 1 <= label <= 12 and score >= args.score_threshold:
                    predictions[condition].append({
                        "image_id": image_id, "category_id": label,
                        "score": score, "box": [float(v) for v in box],
                    })
        if (image_id + 1) % 50 == 0 or image_id + 1 == len(records):
            elapsed = time.time() - started
            print(f"ExDark validation {image_id+1}/{len(records)} | {elapsed/60:.1f} min", flush=True)
        del raw, pred_raw, restored_control, pred_control, restored_detection, pred_detection, image_tensor

    # Save the expensive predictions before COCO metric aggregation.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "exdark_official_validation_predictions.json"
    cache = {
        "dataset": "ExDark", "split": "official_validation", "num_images": len(records),
        "num_ground_truth_boxes": expected_boxes, "max_side": args.exdark_max_side,
        "score_threshold": args.score_threshold,
        "image_sizes": image_sizes,
        "ground_truth": [
            {str(class_id): boxes.tolist() for class_id, boxes in item.items()}
            for item in gt_by_image
        ],
        "predictions": predictions,
    }
    cache_path.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
    results = coco_metrics(gt_by_image, predictions, image_sizes)
    metrics = {
        "dataset": "ExDark", "split": "official_validation",
        "num_images": len(records), "num_ground_truth_boxes": expected_boxes,
        "teacher": str(args.teacher_checkpoint),
        "teacher_training_source": "ExDark teacher_train partition; validation is official ExDark validation",
        "max_side": args.exdark_max_side,
        "score_threshold": args.score_threshold,
        "metric_protocol": "pycocotools COCOeval bbox; IoU=.50:.05:.95; 101 recall thresholds; maxDets=[1,10,100]",
        "results": results,
        "prediction_cache": str(cache_path),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    output = args.output_dir / "exdark_official_validation.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(args.output_dir / "balanced_v2_summary.csv", "ExDark", results)
    print("\nExDark official validation:")
    for name, values in results.items():
        print(f"  {name:18s} mAP50={values['mAP50']:.4f} mAP50:95={values['mAP50_95']:.4f} mAP75={values['mAP75']:.4f}")
    print(f"Saved: {output}", flush=True)
    return metrics


def write_summary_csv(path: Path, dataset: str, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["dataset", "condition", "PSNR", "SSIM", "LPIPS", "mAP50", "mAP50_95", "mAP75"]
    existing_rows: list[list[str]] = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            old_header = next(reader, None)
            if old_header == header:
                existing_rows = [row for row in reader if row and row[0] != dataset]
    new_rows: list[list[Any]] = []
    for condition, values in results.items():
        if dataset == "LOLBlur":
            new_rows.append([dataset, condition, values["psnr"], values["ssim"], values["lpips"], "", "", ""])
        else:
            new_rows.append([dataset, condition, "", "", "", values["mAP50"], values["mAP50_95"], values["mAP75"]])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(existing_rows)
        writer.writerows(new_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "lolblur", "exdark"], default="all")
    parser.add_argument("--lolblur-root", type=Path, default=Path("data/datasets/LOLBlur/train"))
    parser.add_argument("--lolblur-manifest", type=Path, default=Path("data/manifests/balanced_v2/lolblur_internal_val.json"))
    parser.add_argument("--exdark-root", type=Path, default=Path("data/datasets/ExDark"))
    parser.add_argument("--exdark-manifest", type=Path, default=Path("data/manifests/balanced_v2/exdark_official_val.json"))
    parser.add_argument("--control-checkpoint", type=Path, default=Path("models/balanced_v2_pilots/DarkIR_v2_control_pilot_200.pt"))
    parser.add_argument("--detection-checkpoint", type=Path, default=Path("models/balanced_v2_pilots/DarkIR_v2_detection_pilot_200.pt"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("models/exdark_teacher_v2/best.pt"))
    parser.add_argument("--exdark-max-side", type=int, default=640, help="Cap ExDark long side; 640 matches the V2 detector-guidance training input size; 0 keeps native size")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=Path("results/balanced_v2_validation"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.exdark_max_side < 0 or not (0.0 <= args.score_threshold <= 1.0):
        raise ValueError("exdark-max-side must be nonnegative and score-threshold must be in [0,1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run validation on the RTX 4000 Ada GPU")
    device = torch.device("cuda:0")
    print(f"Balanced V2 validation | GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Control checkpoint: {args.control_checkpoint}", flush=True)
    print(f"Detection checkpoint: {args.detection_checkpoint}", flush=True)

    if args.stage in ("all", "lolblur"):
        validate_lolblur(args, device)
    if args.stage in ("all", "exdark"):
        validate_exdark(args, device)
    print("\nBALANCED V2 VALIDATION COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
