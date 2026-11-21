#!/usr/bin/env python3
# Team responsibility: Sanjay Naidu -- ExDark Faster R-CNN teacher training.
"""Fine-tune a 12-class ExDark Faster R-CNN teacher for Balanced V2.

Run from the DarkIR repository root. Uses the generated manifests in
`data/manifests/balanced_v2/`. This script is local-only: it does not modify
repository source files or upload anything.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
EXDARK_TO_COCO_NAME = {
    "bicycle": "bicycle", "boat": "boat", "bottle": "bottle", "bus": "bus",
    "car": "car", "cat": "cat", "chair": "chair", "cup": "cup", "dog": "dog",
    "motorbike": "motorcycle", "people": "person", "table": "dining table",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path, split: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != "ExDark" or payload.get("format_version") != 1:
        raise ValueError(f"Unsupported ExDark manifest format: {path}")
    if payload.get("split") != split:
        raise ValueError(f"Expected split={split!r} in {path}, got {payload.get('split')!r}")
    if payload.get("classes") != EXDARK_CLASSES:
        raise ValueError(f"Class order mismatch in {path}; refusing to train with incorrect IDs")
    if payload.get("label_ids") != list(range(1, 13)):
        raise ValueError(f"Expected contiguous labels 1..12 in {path}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest has no records: {path}")
    for i, rec in enumerate(records):
        if not rec.get("image") or not rec.get("boxes"):
            raise ValueError(f"Manifest record {i} has no image/boxes in {path}")
        for row in rec["boxes"]:
            if len(row) != 5 or not 1 <= int(row[4]) <= 12:
                raise ValueError(f"Invalid [x1,y1,x2,y2,label] box in {path}, record {i}: {row}")
            x1, y1, x2, y2 = map(float, row[:4])
            if not (x2 > x1 and y2 > y1):
                raise ValueError(f"Degenerate box in {path}, record {i}: {row}")
    return payload


class ExDarkManifestDataset(Dataset):
    def __init__(self, manifest: dict[str, Any], root: Path, augment: bool):
        self.records = manifest["records"]
        self.root = root
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.root / record["image"]
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            width, height = image.size
            arr = np.asarray(image, dtype=np.uint8).copy()
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)

        rows = record["boxes"]
        boxes = torch.tensor([r[:4] for r in rows], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([int(r[4]) for r in rows], dtype=torch.int64)
        if (boxes[:, 0] < 0).any() or (boxes[:, 1] < 0).any() or \
           (boxes[:, 2] > width + 1e-3).any() or (boxes[:, 3] > height + 1e-3).any():
            raise RuntimeError(f"Out-of-bounds box in manifest for {image_path}")

        if self.augment and random.random() < 0.5:
            image_tensor = image_tensor.flip(-1)
            old_x1 = boxes[:, 0].clone()
            boxes[:, 0] = width - boxes[:, 2]
            boxes[:, 2] = width - old_x1

        target = {
            "boxes": boxes.contiguous(),
            "labels": labels.contiguous(),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).contiguous(),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return image_tensor.contiguous(), target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_teacher() -> tuple[torch.nn.Module, dict[str, int]]:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    coco_categories = weights.meta["categories"]
    coco_id_by_exdark = {}
    for exdark_name in EXDARK_CLASSES:
        coco_name = EXDARK_TO_COCO_NAME[exdark_name.lower()]
        if coco_name not in coco_categories:
            raise RuntimeError(f"Torchvision COCO weights do not contain category {coco_name!r}")
        coco_id_by_exdark[exdark_name.lower()] = coco_categories.index(coco_name)

    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    old_predictor = model.roi_heads.box_predictor
    in_features = old_predictor.cls_score.in_features
    new_predictor = FastRCNNPredictor(in_features, num_classes=13)

    # Preserve pretrained COCO classification and class-specific box-regression
    # rows for each matching ExDark category. Output label 0 remains background.
    with torch.no_grad():
        new_predictor.cls_score.weight[0].copy_(old_predictor.cls_score.weight[0])
        new_predictor.cls_score.bias[0].copy_(old_predictor.cls_score.bias[0])
        new_predictor.bbox_pred.weight[0:4].copy_(old_predictor.bbox_pred.weight[0:4])
        new_predictor.bbox_pred.bias[0:4].copy_(old_predictor.bbox_pred.bias[0:4])
        for target_id, exdark_name in enumerate(EXDARK_CLASSES, start=1):
            source_id = coco_id_by_exdark[exdark_name.lower()]
            new_predictor.cls_score.weight[target_id].copy_(old_predictor.cls_score.weight[source_id])
            new_predictor.cls_score.bias[target_id].copy_(old_predictor.cls_score.bias[source_id])
            src = slice(source_id * 4, source_id * 4 + 4)
            dst = slice(target_id * 4, target_id * 4 + 4)
            new_predictor.bbox_pred.weight[dst].copy_(old_predictor.bbox_pred.weight[src])
            new_predictor.bbox_pred.bias[dst].copy_(old_predictor.bbox_pred.bias[src])

    model.roi_heads.box_predictor = new_predictor
    # Keep weak detections for validation AP; COCOeval ranks by confidence.
    model.roi_heads.score_thresh = 0.001
    model.roi_heads.detections_per_img = 100
    return model, coco_id_by_exdark


def move_targets_to_device(targets, device):
    return [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]


def evaluate_coco(model, loader, manifest, device, amp: bool) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("pycocotools is required. Install with: pip install pycocotools") from exc

    images_json, annotations_json, predictions = [], [], []
    ann_id = 1
    # Ground truth is sourced from the same validated manifest used by the Dataset.
    for image_id, record in enumerate(manifest["records"]):
        images_json.append({
            "id": image_id, "file_name": record["image"],
            "width": int(record["width"]), "height": int(record["height"]),
        })
        for row in record["boxes"]:
            x1, y1, x2, y2, label = row
            w, h = float(x2) - float(x1), float(y2) - float(y1)
            annotations_json.append({
                "id": ann_id, "image_id": image_id, "category_id": int(label),
                "bbox": [float(x1), float(y1), w, h],
                "area": w * h, "iscrowd": 0,
            })
            ann_id += 1

    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            image_ids = [int(t["image_id"].item()) for t in targets]
            images = [im.to(device, non_blocking=True) for im in images]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                outputs = model(images)
            for image_id, output in zip(image_ids, outputs):
                boxes = output["boxes"].float().cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].float().cpu().numpy()
                for box, label, score in zip(boxes, labels, scores):
                    if int(label) < 1 or int(label) > 12:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box]
                    w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    predictions.append({
                        "image_id": image_id, "category_id": int(label),
                        "bbox": [x1, y1, w, h], "score": float(score),
                    })

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {"description": "ExDark teacher validation"},
        "images": images_json,
        "annotations": annotations_json,
        "categories": [
            {"id": i + 1, "name": EXDARK_CLASSES[i].lower()} for i in range(12)
        ],
    }
    coco_gt.createIndex()
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": images_json,
            "categories": coco_gt.dataset["categories"],
            "annotations": [],
        }
        coco_dt.createIndex()

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [x["id"] for x in images_json]
    evaluator.params.catIds = list(range(1, 13))
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    per_class = {}
    precision = evaluator.eval["precision"]
    for class_index, class_name in enumerate(EXDARK_CLASSES):
        p_all = precision[:, :, class_index, 0, 2]
        p50 = precision[0, :, class_index, 0, 2]
        p_all = p_all[p_all > -1]
        p50 = p50[p50 > -1]
        per_class[class_name] = {
            "AP50": float(p50.mean()) if p50.size else None,
            "AP50_95": float(p_all.mean()) if p_all.size else None,
        }

    return {
        "map50_95": float(evaluator.stats[0]),
        "map50": float(evaluator.stats[1]),
        "map75": float(evaluator.stats[2]),
        "num_predictions": len(predictions),
        "per_class": per_class,
        "metric_protocol": "pycocotools COCOeval bbox, IoU=.50:.05:.95, maxDets=100",
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int,
                    best_map: float, args, coco_map: dict[str, int], metrics: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_map50_95": best_map,
        "class_names": EXDARK_CLASSES,
        "num_classes_with_background": 13,
        "exdark_to_coco_category_index": coco_map,
        "metrics": metrics,
        "args": vars(args),
    }, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/ExDark"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests/balanced_v2"))
    parser.add_argument("--train-manifest", default="exdark_teacher_train.json")
    parser.add_argument("--val-manifest", default="exdark_teacher_val.json")
    parser.add_argument("--output-dir", type=Path, default=Path("models/exdark_teacher_v2"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=5e-4)
    parser.add_argument("--head-lr", type=float, default=2.5e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this teacher training script on the lab GPU.")
    if not args.amp and args.batch_size > 1:
        print("WARNING: AMP is disabled. If you hit CUDA OOM, use --batch-size 1 or enable AMP.")

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"PyTorch: {torch.__version__} | torchvision: {__import__('torchvision').__version__}", flush=True)

    train_manifest = load_manifest(args.manifest_dir / args.train_manifest, "teacher_train")
    val_manifest = load_manifest(args.manifest_dir / args.val_manifest, "teacher_validation")
    print(
        f"Teacher train: {len(train_manifest['records'])} images / {train_manifest['num_boxes']} boxes | "
        f"teacher val: {len(val_manifest['records'])} images / {val_manifest['num_boxes']} boxes",
        flush=True,
    )

    train_ds = ExDarkManifestDataset(train_manifest, args.root, augment=True)
    val_ds = ExDarkManifestDataset(val_manifest, args.root, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, collate_fn=detection_collate,
    )

    model, coco_map = build_teacher()
    model.to(device)
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    milestones = sorted(set([max(1, int(args.epochs * 0.60)), max(1, int(args.epochs * 0.82))]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    start_epoch = 1
    best_map = -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("class_names") != EXDARK_CLASSES:
            raise ValueError("Resume checkpoint class order differs from current ExDark label mapping")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint.get("best_map50_95", -1.0))
        print(f"Resuming from {args.resume} at epoch {start_epoch}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep run metadata in the repository's ignored root results/*.json area.
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "exdark_teacher_v2_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = defaultdict(float)
        seen = 0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = move_targets_to_device(targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                loss_dict = model(images, targets)
                total_loss = sum(loss_dict.values())
            if not torch.isfinite(total_loss):
                parts = {k: float(v.detach().float().cpu()) for k, v in loss_dict.items()}
                raise RuntimeError(f"Non-finite detector loss at epoch={epoch}, step={step}: {parts}")
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            batch_n = len(images)
            seen += batch_n
            running["total"] += float(total_loss.detach().float().cpu()) * batch_n
            for key, value in loss_dict.items():
                running[key] += float(value.detach().float().cpu()) * batch_n
            if step == 1 or step % args.log_every == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={float(total_loss.detach().float().cpu()):.4f} "
                    f"lr_backbone={optimizer.param_groups[0]['lr']:.2e} "
                    f"lr_heads={optimizer.param_groups[1]['lr']:.2e}",
                    flush=True,
                )

        scheduler.step()
        train_metrics = {f"train_{k}": v / max(seen, 1) for k, v in running.items()}
        print(f"Epoch {epoch} training summary: {train_metrics}", flush=True)
        print(f"Running teacher validation ({len(val_ds)} images)...", flush=True)
        val_metrics = evaluate_coco(model, val_loader, val_manifest, device, args.amp)
        epoch_record = {
            "epoch": epoch,
            **train_metrics,
            "val_map50": val_metrics["map50"],
            "val_map50_95": val_metrics["map50_95"],
            "val_map75": val_metrics["map75"],
            "val_num_predictions": val_metrics["num_predictions"],
            "val_per_class": val_metrics["per_class"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(epoch_record)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

        is_new_best = val_metrics["map50_95"] > best_map
        if is_new_best:
            best_map = val_metrics["map50_95"]
            save_checkpoint(
                args.output_dir / "best.pt", model, optimizer, scheduler, scaler,
                epoch, best_map, args, coco_map, val_metrics,
            )
            print(f"New best teacher: mAP50:95={best_map:.4f}", flush=True)
        save_checkpoint(
            args.output_dir / "last.pt", model, optimizer, scheduler, scaler,
            epoch, best_map, args, coco_map, val_metrics,
        )
        print(
            f"EPOCH {epoch} COMPLETE | val mAP50={val_metrics['map50']:.4f} "
            f"mAP50:95={val_metrics['map50_95']:.4f} mAP75={val_metrics['map75']:.4f} "
            f"time={(time.time() - epoch_start)/60:.1f} min",
            flush=True,
        )

    print(f"TEACHER TRAINING FINISHED. Best validation mAP50:95={best_map:.4f}", flush=True)
    print(f"Best checkpoint: {args.output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
