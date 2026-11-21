#!/usr/bin/env python3
# Team responsibility: Bhanu Prakash -- evaluator implementation; Sanjay Naidu -- protocol review.
"""Audited ExDark detection evaluation for DarkIR variants using standard COCOeval.

Uses the official ExDark split and bbox format, maps its 12 labels to matching COCO
classes, configures Faster R-CNN score filtering consistently, and uses pycocotools
COCOeval for bbox AP. The generated prediction cache is saved alongside the metrics.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from archs.DarkIR import DarkIR

EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
EXDARK_TO_COCO_NAME = {
    "bicycle": "bicycle", "boat": "boat", "bottle": "bottle", "bus": "bus",
    "car": "car", "cat": "cat", "chair": "chair", "cup": "cup", "dog": "dog",
    "motorbike": "motorcycle", "people": "person", "table": "dining table",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_darkir(checkpoint_path, device):
    model = DarkIR(
        img_channel=3, width=32, middle_blk_num_enc=2, middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9], extra_depth_wise=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
    weights = {k.removeprefix("module."): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def parse_split_file(split_file, split_code):
    records = []
    with open(split_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 5:
                continue
            try:
                class_code = int(fields[1])
                fold = int(fields[-1])
            except ValueError:  # header or non-data line
                continue
            if fold != split_code:
                continue
            rel = fields[0].replace("\\", "/")
            class_name = EXDARK_CLASSES[class_code - 1] if 1 <= class_code <= len(EXDARK_CLASSES) else None
            records.append((rel, class_name))
    if not records:
        raise RuntimeError(f"No split={split_code} records found in {split_file}. Check the official imageclasslist.txt format/path.")
    return records


def build_image_index(images_root):
    index = defaultdict(list)
    for p in Path(images_root).rglob("*"):
        # The official ZIP may include macOS AppleDouble sidecar files. These
        # can share image extensions but are not actual images.
        if any(part == "__MACOSX" for part in p.parts) or p.name.startswith("._"):
            continue
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            index[p.name.lower()].append(p)
    return index


def resolve_image(rel, class_name, images_root, image_index):
    root = Path(images_root)
    rel_path = Path(rel)
    candidates = [root / rel_path]
    if class_name:
        candidates.append(root / class_name / rel_path.name)
    # Some archives include an extra Dataset/ directory; the supplied images_root may be its parent.
    candidates.extend([root / "Dataset" / rel_path, root / "dataset" / rel_path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = image_index.get(rel_path.name.lower(), [])
    if class_name:
        same_class = [p for p in matches if p.parent.name.lower() == class_name.lower()]
        if len(same_class) == 1:
            return same_class[0]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous image basename {rel}; set --images-root to the directory containing class folders.")
    raise FileNotFoundError(f"Image not found for split entry {rel} under {images_root}")


def build_annotation_index(annotations_root):
    index = defaultdict(list)
    for p in Path(annotations_root).rglob("*.txt"):
        if any(part == "__MACOSX" for part in p.parts) or p.name.startswith("._"):
            continue
        if p.name.lower() != "imageclasslist.txt":
            # Official ExDark annotations are named IMAGE.png.txt, so the
            # lookup key must retain the image extension (IMAGE.png).
            key = p.name[:-4].lower() if p.name.lower().endswith(".txt") else p.stem.lower()
            index[key].append(p)
    return index


def resolve_annotation(image_path, class_name, annotations_root, annotation_index):
    root = Path(annotations_root)
    image_name = image_path.name.lower()
    annotation_name = f"{image_path.name}.txt"
    if class_name:
        # Official archive structure: Groundtruth/ExDark_Annno/CLASS/IMAGE.png.txt
        candidates = [
            root / "ExDark_Annno" / class_name / annotation_name,
            root / "exdark_annno" / class_name / annotation_name,
            root / class_name / annotation_name,
            root / "Groundtruth" / class_name / annotation_name,
            root / "groundtruth" / class_name / annotation_name,
        ]
        for exact in candidates:
            if exact.is_file():
                return exact
    matches = annotation_index.get(image_name, [])
    if class_name:
        same_class = [p for p in matches if p.parent.name.lower() == class_name.lower()]
        if len(same_class) == 1:
            return same_class[0]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous annotation for {image_path.name}; check --annotations-root.")
    raise FileNotFoundError(f"Annotation not found for {image_path.name} under {annotations_root}")


def norm_class_name(name):
    key = name.strip().lower().replace("_", " ")
    aliases = {"motor bike": "motorbike", "motorcycle": "motorbike", "person": "people", "dining table": "table"}
    return aliases.get(key, key)


def parse_gt_file(path, width, height, coco_label_by_exdark_name, scale_x=1.0, scale_y=1.0):
    """Parse ExDark's `class left top width height ...` annotation rows.

    Official files may begin with a `% bbGt version=3` metadata line. The README
    describes a 16-character annotation-tool metadata prefix; to support both the
    distributed file format (header on a separate line) and compatible exports,
    each row is tried as-is first and with the 16-character prefix removed second.
    """
    gt = defaultdict(list)
    valid_classes = set(coco_label_by_exdark_name)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith(("%", "#")):
                continue
            candidates = [line]
            if len(raw_line) > 16:
                candidates.append(raw_line[16:].strip())
            for candidate in candidates:
                fields = candidate.split()
                if len(fields) < 5:
                    continue
                cls = norm_class_name(fields[0])
                if cls not in valid_classes:
                    continue
                try:
                    x, y, w, h = map(float, fields[1:5])
                except ValueError:
                    continue
                # ExDark uses [left, top, width, height]; convert to xyxy.
                x1 = max(0.0, min(width, x * scale_x))
                y1 = max(0.0, min(height, y * scale_y))
                x2 = max(0.0, min(width, (x + w) * scale_x))
                y2 = max(0.0, min(height, (y + h) * scale_y))
                if x2 > x1 and y2 > y1:
                    gt[coco_label_by_exdark_name[cls]].append([x1, y1, x2, y2])
                break
    return {k: np.asarray(v, dtype=np.float32).reshape(-1, 4) for k, v in gt.items()}


def box_iou_one(box, boxes):
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def ap_for_class(predictions, gt_by_image, category_id, iou_threshold):
    npos = sum(len(record.get(category_id, [])) for record in gt_by_image)
    if npos == 0:
        return float("nan")
    dets = [p for p in predictions if p["category_id"] == category_id]
    dets.sort(key=lambda p: p["score"], reverse=True)
    matched = {i: np.zeros(len(record.get(category_id, [])), dtype=bool) for i, record in enumerate(gt_by_image)}
    tp = np.zeros(len(dets), dtype=np.float64)
    fp = np.zeros(len(dets), dtype=np.float64)
    for k, det in enumerate(dets):
        image_id = det["image_id"]
        gt_boxes = gt_by_image[image_id].get(category_id, np.zeros((0, 4), dtype=np.float32))
        if len(gt_boxes) == 0:
            fp[k] = 1
            continue
        ious = box_iou_one(np.asarray(det["box"], dtype=np.float32), gt_boxes)
        # Match to the highest-IoU *unmatched* GT. If we simply choose the
        # overall best IoU and then discover that it was already matched, a
        # valid detection of a second nearby same-class object is incorrectly
        # counted as a false positive. COCO-style matching searches available
        # ground truths before marking a detection as a duplicate.
        available = ~matched[image_id]
        if len(ious) and available.any():
            eligible_ious = np.where(available, ious, -1.0)
            best = int(eligible_ious.argmax())
            if eligible_ious[best] >= iou_threshold:
                tp[k] = 1
                matched[image_id][best] = True
            else:
                fp[k] = 1
        else:
            fp[k] = 1
    if len(dets) == 0:
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(npos, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    # VOC/COCO-style 101-point interpolated AP.
    samples = np.linspace(0, 1, 101)
    interp = []
    for r in samples:
        eligible = precision[recall >= r]
        interp.append(float(eligible.max()) if eligible.size else 0.0)
    return float(np.mean(interp))


def summarize_map(predictions, gt_by_image, exdark_to_coco_id):
    valid_ids = [coco_id for coco_id in exdark_to_coco_id.values()
                 if any(len(rec.get(coco_id, [])) for rec in gt_by_image)]
    per_class = {}
    for name, cid in exdark_to_coco_id.items():
        ap50 = ap_for_class(predictions, gt_by_image, cid, 0.50)
        aps = [ap_for_class(predictions, gt_by_image, cid, t) for t in np.arange(0.50, 0.951, 0.05)]
        aps = [v for v in aps if np.isfinite(v)]
        per_class[name] = {"AP50": ap50 if np.isfinite(ap50) else None,
                           "AP50_95": float(np.mean(aps)) if aps else None}
    all50 = [per_class[n]["AP50"] for n in per_class if per_class[n]["AP50"] is not None]
    all5095 = [per_class[n]["AP50_95"] for n in per_class if per_class[n]["AP50_95"] is not None]
    return {
        "mAP50": float(np.mean(all50)) if all50 else None,
        "mAP50_95": float(np.mean(all5095)) if all5095 else None,
        "classes_with_ground_truth": len(valid_ids),
        "per_class": per_class,
    }


def evaluate_with_cocoeval(prepared, gt_by_image, predictions_by_condition, exdark_to_coco_id):
    """Evaluate saved xyxy predictions using official pycocotools COCOeval."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for standard COCO metrics. Install it with: "
            "python -m pip install pycocotools"
        ) from exc

    image_records = []
    for image_id, (path, width, height) in enumerate(prepared):
        image_records.append({
            "id": image_id, "file_name": str(path),
            "width": int(width), "height": int(height),
        })

    category_ids = sorted(set(exdark_to_coco_id.values()))
    coco_to_name = {cid: EXDARK_TO_COCO_NAME[name] for name, cid in exdark_to_coco_id.items()}
    categories = [{"id": cid, "name": coco_to_name[cid]} for cid in category_ids]

    gt_annotations = []
    ann_id = 1
    for image_id, image_gt in enumerate(gt_by_image):
        for category_id, boxes in image_gt.items():
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box]
                w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                gt_annotations.append({
                    "id": ann_id, "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
                })
                ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {"description": "ExDark split mapped to COCO categories"},
        "images": image_records,
        "annotations": gt_annotations,
        "categories": categories,
    }
    coco_gt.createIndex()

    summaries = {}
    for condition, preds in predictions_by_condition.items():
        result_rows = []
        for pred in preds:
            x1, y1, x2, y2 = [float(v) for v in pred["box"]]
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            result_rows.append({
                "image_id": int(pred["image_id"]),
                "category_id": int(pred["category_id"]),
                "bbox": [x1, y1, w, h],
                "score": float(pred["score"]),
            })
        if result_rows:
            coco_dt = coco_gt.loadRes(result_rows)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {"images": image_records, "categories": categories, "annotations": []}
            coco_dt.createIndex()

        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = [row["id"] for row in image_records]
        evaluator.params.catIds = category_ids
        evaluator.params.iouThrs = np.linspace(0.50, 0.95, 10)
        evaluator.params.recThrs = np.linspace(0.0, 1.0, 101)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        precision = evaluator.eval["precision"]  # T x R x K x A x M
        cat_index = {cid: i for i, cid in enumerate(evaluator.params.catIds)}
        per_class = {}
        ap50_values, ap_values = [], []
        for exdark_name, category_id in exdark_to_coco_id.items():
            k = cat_index[category_id]
            p_all = precision[:, :, k, 0, 2]  # area=all, maxDets=100
            valid_all = p_all[p_all > -1]
            p50 = precision[0, :, k, 0, 2]
            valid50 = p50[p50 > -1]
            ap = float(valid_all.mean()) if valid_all.size else None
            ap50 = float(valid50.mean()) if valid50.size else None
            per_class[exdark_name] = {"AP50": ap50, "AP50_95": ap}
            if ap50 is not None:
                ap50_values.append(ap50)
            if ap is not None:
                ap_values.append(ap)

        summaries[condition] = {
            # stats[0] and stats[1] are standard COCO AP@[.50:.95] and AP@.50.
            "mAP50": float(evaluator.stats[1]),
            "mAP50_95": float(evaluator.stats[0]),
            "mAP75": float(evaluator.stats[2]),
            "per_class": per_class,
            "num_predictions_after_filter": len(result_rows),
            "metric_protocol": "pycocotools COCOeval bbox, IoU=.50:.05:.95, 101 recall thresholds, maxDets=100",
        }
        print(
            f"{condition}: COCO mAP50={summaries[condition]['mAP50']:.4f} | "
            f"COCO mAP50:95={summaries[condition]['mAP50_95']:.4f} | "
            f"mAP75={summaries[condition]['mAP75']:.4f}", flush=True
        )
    return summaries, image_records, gt_annotations


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-root", required=True, help="Directory containing ExDark class folders (or their parent)")
    ap.add_argument("--annotations-root", required=True, help="Extracted official ExDark Groundtruth directory")
    ap.add_argument("--split-file", required=True, help="Official imageclasslist.txt")
    ap.add_argument("--split", choices=["val", "test"], default="val", help="Use val=2 for model decisions; test=3 only for final evaluation")
    ap.add_argument("--pretrained", default="models/DarkIR_384.pt")
    ap.add_argument("--control", default="models/DarkIR_control_1000.pt")
    ap.add_argument("--task", default="models/DarkIR_task_1000.pt")
    ap.add_argument("--max-side", type=int, default=1280, help="Resize long side to at most this value; boxes are scaled consistently. Set 0 to keep original size.")
    ap.add_argument("--score-threshold", type=float, default=0.001)
    ap.add_argument("--output", default="results/exdark_val_detection.json")
    ap.add_argument("--preflight-only", action="store_true",
                    help="Validate split, image paths, annotations, and parsed boxes without loading models or running inference")
    ap.add_argument("--reuse-predictions-cache", default=None,
                    help="Recompute COCO metrics from a previous *_predictions.json cache without GPU inference")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split_code = 2 if args.split == "val" else 3
    records = parse_split_file(args.split_file, split_code)
    if args.split == "val" and len(records) != 1800:
        raise RuntimeError(f"Official ExDark validation split should contain 1800 records, found {len(records)}. Check imageclasslist.txt and its extraction.")
    if args.split == "test" and len(records) != 2563:
        raise RuntimeError(f"Official ExDark test split should contain 2563 records, found {len(records)}. Check imageclasslist.txt and its extraction.")

    # Resolve every image and annotation and parse GTs before loading the GPU models.
    # This fails fast if archives were nested unexpectedly or annotation parsing is wrong.
    image_index = build_image_index(args.images_root)
    annotation_index = build_annotation_index(args.annotations_root)
    print(f"Official ExDark {args.split} split records: {len(records)}", flush=True)
    # Category IDs will be filled from the pretrained detector's torchvision metadata.
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    category_names = weights.meta["categories"]
    exdark_to_coco_id = {}
    for exdark_class in EXDARK_CLASSES:
        coco_name = EXDARK_TO_COCO_NAME[exdark_class.lower()]
        if coco_name not in category_names:
            raise RuntimeError(f"COCO weights metadata has no category named {coco_name!r}")
        exdark_to_coco_id[exdark_class.lower()] = category_names.index(coco_name)

    prepared = []
    gt_by_image = []
    total_gt_boxes = 0
    gt_class_counts = defaultdict(int)
    empty_gt_images = []
    for rel, class_name in records:
        path = resolve_image(rel, class_name, args.images_root, image_index)
        annotation_path = resolve_annotation(path, class_name, args.annotations_root, annotation_index)
        with Image.open(path) as im:
            orig_w, orig_h = im.size
        if args.max_side and max(orig_w, orig_h) > args.max_side:
            scale = args.max_side / max(orig_w, orig_h)
            new_w, new_h = round(orig_w * scale), round(orig_h * scale)
        else:
            new_w, new_h = orig_w, orig_h
        gt = parse_gt_file(annotation_path, new_w, new_h, exdark_to_coco_id,
                           scale_x=new_w / orig_w, scale_y=new_h / orig_h)
        if not gt:
            empty_gt_images.append((rel, str(annotation_path)))
        for cid, boxes in gt.items():
            gt_class_counts[cid] += len(boxes)
            total_gt_boxes += len(boxes)
        prepared.append((path, new_w, new_h))
        gt_by_image.append(gt)

    if empty_gt_images:
        details = "\n".join(f"  {rel} -> {ann}" for rel, ann in empty_gt_images[:10])
        raise RuntimeError(
            f"Found {len(empty_gt_images)} split images with no parsed ground-truth boxes. "
            f"Refusing to compute AP; inspect these annotation files:\n{details}"
        )
    if total_gt_boxes == 0:
        raise RuntimeError("Parsed zero ground-truth boxes. Check official annotation format and paths.")
    print(f"Preflight passed: {len(prepared)} images, {total_gt_boxes} boxes across {len(gt_class_counts)} mapped classes.", flush=True)
    for name in EXDARK_TO_COCO_NAME:
        cid = exdark_to_coco_id[name]
        print(f"  {name:12s}: {gt_class_counts.get(cid, 0)} boxes")

    if args.preflight_only:
        print("PREFLIGHT ONLY PASSED. No detector weights loaded and no inference run.", flush=True)
        return
    # Fail before spending GPU time if the standard metric package is missing.
    try:
        from pycocotools.coco import COCO as _COCO_CHECK  # noqa: F401
        from pycocotools.cocoeval import COCOeval as _COCOEVAL_CHECK  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required before inference. Install it first with: "
            "python -m pip install pycocotools"
        ) from exc
    if args.reuse_predictions_cache:
        cache_path = Path(args.reuse_predictions_cache)
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("split") != args.split or cache.get("num_images") != len(records):
            raise RuntimeError("Prediction cache split/image count does not match the current preflight dataset")
        cached_prepared = cache.get("prepared", [])
        if len(cached_prepared) != len(prepared):
            raise RuntimeError("Prediction cache image list length differs from current split")
        for i, (item, current) in enumerate(zip(cached_prepared, prepared)):
            path, w, h = current
            if item.get("path") != str(path) or int(item.get("width", -1)) != w or int(item.get("height", -1)) != h:
                raise RuntimeError(f"Prediction cache image metadata differs at split row {i}: {item} vs {current}")
        all_preds = cache.get("predictions")
        expected_conditions = {"original_low_light", "darkir_pretrained", "darkir_control_ft", "darkir_task_aware_ft"}
        if not isinstance(all_preds, dict) or set(all_preds) != expected_conditions:
            raise RuntimeError("Prediction cache is missing one or more expected evaluation conditions")
        coco_summaries, _, _ = evaluate_with_cocoeval(prepared, gt_by_image, all_preds, exdark_to_coco_id)
        result = {
            "dataset": "ExDark", "split": args.split, "split_code": split_code,
            "num_images": len(records), "num_ground_truth_boxes": total_gt_boxes,
            "detector": "torchvision Faster R-CNN ResNet-50 FPN, pretrained on COCO and frozen",
            "metric_protocol": "pycocotools COCOeval bbox, IoU=.50:.05:.95, 101 recall thresholds, maxDets=100",
            "max_side": cache.get("max_side", args.max_side),
            "score_threshold": cache.get("score_threshold", args.score_threshold),
            "class_mapping": {k: EXDARK_TO_COCO_NAME[k] for k in EXDARK_TO_COCO_NAME},
            "results": coco_summaries,
            "reused_prediction_cache": str(cache_path),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"Saved metrics from existing predictions: {output_path}", flush=True)
        return

    if device.type != "cuda":
        raise RuntimeError("CUDA is not available; run on the GPU for a timely evaluation.")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    detector = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None).to(device).eval()
    # Set the threshold inside torchvision too; otherwise its default 0.05
    # filters out low-score detections before our --score-threshold is applied.
    detector.roi_heads.score_thresh = args.score_threshold
    detector.roi_heads.detections_per_img = 100
    for p in detector.parameters():
        p.requires_grad_(False)
    print("Mapped ExDark classes:", {k: (EXDARK_TO_COCO_NAME[k], v) for k, v in exdark_to_coco_id.items()}, flush=True)

    models = {
        "darkir_pretrained": load_darkir(args.pretrained, device),
        "darkir_control_ft": load_darkir(args.control, device),
        "darkir_task_aware_ft": load_darkir(args.task, device),
    }
    conditions = ["original_low_light", *models.keys()]
    all_preds = {condition: [] for condition in conditions}
    start = time.time()
    mapped_label_ids = set(exdark_to_coco_id.values())

    for image_id, (path, new_w, new_h) in enumerate(prepared):
        with Image.open(path) as pil:
            pil = pil.convert("RGB")
            if pil.size != (new_w, new_h):
                pil = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
            x = to_tensor(pil)
        x_gpu = x.to(device)
        with torch.inference_mode():
            condition_images = {"original_low_light": x_gpu}
            for name, model in models.items():
                condition_images[name] = model(x_gpu.unsqueeze(0)).squeeze(0).clamp(0, 1)
            for condition in conditions:
                pred = detector([condition_images[condition]])[0]
                boxes = pred["boxes"].detach().cpu().numpy()
                labels = pred["labels"].detach().cpu().numpy()
                scores = pred["scores"].detach().cpu().numpy()
                for box, label, score in zip(boxes, labels, scores):
                    if score < args.score_threshold or int(label) not in mapped_label_ids:
                        continue
                    all_preds[condition].append({
                        "image_id": image_id, "category_id": int(label),
                        "score": float(score), "box": [float(v) for v in box],
                    })
        if (image_id + 1) % 50 == 0 or image_id + 1 == len(prepared):
            elapsed = time.time() - start
            print(f"processed {image_id+1}/{len(prepared)} images | {elapsed/60:.1f} min", flush=True)

    # Save raw predictions and parsed GT before metric aggregation. If COCOeval
    # raises unexpectedly, reruns of metric calculation need not repeat GPU inference.
    output_base = Path(args.output)
    pred_cache_path = output_base.with_name(output_base.stem + "_predictions.json")
    cache = {
        "dataset": "ExDark", "split": args.split, "num_images": len(records),
        "prepared": [
            {"path": str(path), "width": int(width), "height": int(height)}
            for path, width, height in prepared
        ],
        "category_names": {str(v): EXDARK_TO_COCO_NAME[k] for k, v in exdark_to_coco_id.items()},
        "gt_by_image": [
            {str(int(cid)): boxes.tolist() for cid, boxes in image_gt.items()}
            for image_gt in gt_by_image
        ],
        "predictions": all_preds,
        "score_threshold": args.score_threshold,
        "max_side": args.max_side,
    }
    pred_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, separators=(",", ":"))
    print(f"Saved raw predictions cache before metric aggregation: {pred_cache_path}", flush=True)

    coco_summaries, image_records, gt_annotations = evaluate_with_cocoeval(
        prepared, gt_by_image, all_preds, exdark_to_coco_id
    )
    result = {
        "dataset": "ExDark",
        "split": args.split,
        "split_code": split_code,
        "num_images": len(records),
        "num_ground_truth_boxes": total_gt_boxes,
        "detector": "torchvision Faster R-CNN ResNet-50 FPN, pretrained on COCO and frozen",
        "metric_protocol": "pycocotools COCOeval bbox, IoU=.50:.05:.95, 101 recall thresholds, maxDets=100",
        "max_side": args.max_side,
        "score_threshold": args.score_threshold,
        "class_mapping": {k: EXDARK_TO_COCO_NAME[k] for k in EXDARK_TO_COCO_NAME},
        "results": coco_summaries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved metrics: {output_path}")
    print(f"Saved predictions cache: {pred_cache_path}")
    print(f"Total elapsed: {(time.time()-start)/60:.1f} min")
    print("NOTE: zero-shot COCO-detector scores on ExDark; compare relative differences under this fixed protocol, not against ExDark-trained detector benchmarks.")


if __name__ == "__main__":
    main()
