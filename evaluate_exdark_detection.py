#!/usr/bin/env python3
# Team responsibility: Bhanu Prakash -- evaluator implementation; Sanjay Naidu -- protocol review.
"""Compare a frozen COCO Faster R-CNN on ExDark low-light images and DarkIR variants.

Default split is ExDark validation (split code 2). Keep the ExDark test split (code 3)
untouched until final evaluation. Outputs VOC-style 101-point AP@0.50 and mean AP@0.50:0.95
for the 12 ExDark categories that can be mapped to COCO categories.
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
    """Parse official ExDark annotation files.

    The official README says the first 16 characters on annotation records are
    annotation-tool metadata; actual records start with the object class name.
    We try that documented offset first, then fall back to the full line for
    compatible reorganized copies. Only known class names are accepted.
    """
    gt = defaultdict(list)
    valid_classes = set(coco_label_by_exdark_name)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Try official 16-character prefix removal, then no-prefix format.
            candidates = []
            if len(line) > 16:
                candidates.append(line[16:].strip())
            candidates.append(line.strip())
            parsed = False
            for candidate in candidates:
                fields = candidate.split()
                if len(fields) < 5 or fields[0].startswith(("%", "#")):
                    continue
                cls = norm_class_name(fields[0])
                if cls not in valid_classes:
                    continue
                try:
                    x, y, w, h = map(float, fields[1:5])
                except ValueError:
                    continue
                # ExDark documents [left, top, width, height] coordinates.
                x1 = max(0.0, min(width, x * scale_x))
                y1 = max(0.0, min(height, y * scale_y))
                x2 = max(0.0, min(width, (x + w) * scale_x))
                y2 = max(0.0, min(height, (y + h) * scale_y))
                if x2 > x1 and y2 > y1:
                    gt[coco_label_by_exdark_name[cls]].append([x1, y1, x2, y2])
                parsed = True
                break
            # Non-object metadata/header rows are ignored.
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
        best = int(ious.argmax()) if len(ious) else -1
        if best >= 0 and ious[best] >= iou_threshold and not matched[image_id][best]:
            tp[k] = 1
            matched[image_id][best] = True
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
            # Preserve empty-image support, but count it explicitly for diagnostics.
            pass
        for cid, boxes in gt.items():
            gt_class_counts[cid] += len(boxes)
            total_gt_boxes += len(boxes)
        prepared.append((path, new_w, new_h))
        gt_by_image.append(gt)

    if total_gt_boxes == 0:
        raise RuntimeError("Parsed zero ground-truth boxes. Check that official annotation files were extracted and follow the expected bbGt format.")
    print(f"Preflight passed: {len(prepared)} images, {total_gt_boxes} boxes across {len(gt_class_counts)} mapped classes.", flush=True)
    for name in EXDARK_TO_COCO_NAME:
        cid = exdark_to_coco_id[name]
        print(f"  {name:12s}: {gt_class_counts.get(cid, 0)} boxes")

    if args.preflight_only:
        print("PREFLIGHT ONLY PASSED. No detector weights loaded and no inference run.", flush=True)
        return
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available; run on the GPU for a timely evaluation.")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    detector = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None).to(device).eval()
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

    result = {
        "dataset": "ExDark",
        "split": args.split,
        "split_code": split_code,
        "num_images": len(records),
        "num_ground_truth_boxes": total_gt_boxes,
        "detector": "torchvision Faster R-CNN ResNet-50 FPN, pretrained on COCO and frozen",
        "metric_protocol": "class-wise 101-point interpolated AP; mAP@0.50 and mean AP@0.50:0.95",
        "max_side": args.max_side,
        "score_threshold": args.score_threshold,
        "class_mapping": {k: EXDARK_TO_COCO_NAME[k] for k in EXDARK_TO_COCO_NAME},
        "results": {},
    }
    for condition in conditions:
        summary = summarize_map(all_preds[condition], gt_by_image, exdark_to_coco_id)
        result["results"][condition] = summary
        m50 = summary["mAP50"]
        m5095 = summary["mAP50_95"]
        print(f"\n{condition}: mAP50={m50:.4f} | mAP50:95={m5095:.4f}" if m50 is not None and m5095 is not None
              else f"\n{condition}: no valid AP scores; inspect annotation/parser setup")
        for class_name, vals in summary["per_class"].items():
            if vals["AP50"] is not None:
                print(f"  {class_name:12s} AP50={vals['AP50']:.4f} AP50:95={vals['AP50_95']:.4f}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved metrics: {output_path}")
    print(f"Total elapsed: {(time.time()-start)/60:.1f} min")
    print("NOTE: zero-shot COCO-detector scores on ExDark; compare relative differences under this fixed protocol, not against ExDark-trained detector benchmarks.")


if __name__ == "__main__":
    main()
