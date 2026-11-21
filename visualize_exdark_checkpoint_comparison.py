#!/usr/bin/env python3
# Team responsibility: Hemanth Sai -- final-checkpoint qualitative comparison.
"""Render comparable ExDark detector overlays for final DarkIR checkpoints.

Runs the same frozen COCO-pretrained Faster R-CNN on selected official
ExDark-validation images, using the original images and each listed DarkIR
checkpoint as input. This creates a qualitative visualization only; it does
not calculate or replace full-split metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor

import evaluate_exdark_cocoeval as exdark
from evaluate_v2_independent_coco_detector import load_base_module, load_darkir, restore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, default=Path("data/datasets/ExDark/images"))
    parser.add_argument("--annotations-root", type=Path, default=Path("data/datasets/ExDark/Groundtruth"))
    parser.add_argument("--split-file", type=Path, default=Path("data/datasets/ExDark/Groundtruth/imageclasslist.txt"))
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument(
        "--image-ids",
        type=int,
        nargs="+",
        default=[257, 1032, 1513],
        help=(
            "official-validation record IDs; defaults were selected because "
            "the saved V2-Detection 0.10 predictions had more IoU-0.50 "
            "true-positive matches than V2-Control on each example"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("author-kit-CVPR2027-v1/fig/v2_exdark_detection_comparison.png"),
    )
    return parser


def resize_for_evaluation(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    if max_side > 0 and max(width, height) > max_side:
        scale = max_side / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(size, Image.Resampling.BILINEAR)
    return image


def draw_detections(
    image: Image.Image,
    detections: list[dict[str, object]],
    color: str,
    width: int = 3,
) -> None:
    draw = ImageDraw.Draw(image)
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)


def main() -> None:
    args = build_parser().parse_args()
    if args.max_side < 0:
        raise ValueError("--max-side must be nonnegative.")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be in [0,1].")

    root = Path.cwd().resolve()
    if not (root / "evaluate_exdark_cocoeval.py").is_file():
        raise RuntimeError("Run this script from the DarkIR repository root.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the detector/checkpoints.")

    base = load_base_module()
    records = base.parse_split_file(args.split_file, split_code=2)
    if len(records) != 1800:
        raise RuntimeError(f"Expected 1,800 official validation records; found {len(records)}.")
    if len(set(args.image_ids)) != len(args.image_ids):
        raise ValueError("--image-ids must be unique.")
    if any(image_id < 0 or image_id >= len(records) for image_id in args.image_ids):
        raise ValueError("An --image-ids value is outside the official validation split.")

    image_index = base.build_image_index(args.images_root)
    annotation_index = base.build_annotation_index(args.annotations_root)
    selected: list[tuple[int, Path, str, Image.Image, dict[int, list[list[float]]]]] = []
    for image_id in args.image_ids:
        rel_path, class_name = records[image_id]
        image_path = base.resolve_image(rel_path, class_name, args.images_root, image_index)
        annotation_path = base.resolve_annotation(
            image_path, class_name, args.annotations_root, annotation_index
        )
        with Image.open(image_path) as source:
            original_size = source.size
            image = resize_for_evaluation(source.convert("RGB"), args.max_side)
        category_ids = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
        category_map = {
            name.lower(): category_ids.index(base.EXDARK_TO_COCO_NAME[name.lower()])
            for name in base.EXDARK_CLASSES
        }
        gt_arrays = base.parse_gt_file(
            annotation_path,
            image.width,
            image.height,
            category_map,
            scale_x=image.width / original_size[0],
            scale_y=image.height / original_size[1],
        )
        gt = {class_id: boxes.tolist() for class_id, boxes in gt_arrays.items()}
        selected.append((image_id, image_path, class_name, image, gt))

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    detector.roi_heads.score_thresh = args.score_threshold
    detector.roi_heads.detections_per_img = 100
    device = torch.device("cuda:0")
    detector.to(device).eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)

    checkpoint_specs = [
        ("Pretrained DarkIR", Path("models/DarkIR_384.pt")),
        ("V2-Control", Path("models/balanced_v2_5k/DarkIR_v2_control_pilot_5000.pt")),
        ("V2-Det 0.05", Path("models/balanced_v2_detw_005/DarkIR_v2_detection_pilot_5000.pt")),
        ("V2-Det 0.10", Path("models/balanced_v2_5k/DarkIR_v2_detection_pilot_5000.pt")),
        ("V2-Det 0.20", Path("models/balanced_v2_detw_020/DarkIR_v2_detection_pilot_5000.pt")),
    ]
    for label, checkpoint in checkpoint_specs:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint}")

    condition_predictions: dict[str, list[list[dict[str, object]]]] = {
        "Original low-light": [[] for _ in selected],
        **{label: [[] for _ in selected] for label, _ in checkpoint_specs},
    }
    mapped_ids = {
        weights.meta["categories"].index(base.EXDARK_TO_COCO_NAME[name.lower()])
        for name in base.EXDARK_CLASSES
    }

    with torch.inference_mode():
        for row, (_, _, _, image, _) in enumerate(selected):
            tensor = to_tensor(image).to(device)
            output = detector([tensor])[0]
            condition_predictions["Original low-light"][row] = [
                {"box": box.tolist(), "category_id": int(label), "score": float(score)}
                for box, label, score in zip(
                    output["boxes"].cpu(), output["labels"].cpu(), output["scores"].cpu()
                )
                if int(label) in mapped_ids and float(score) >= args.score_threshold
            ]

        for label, checkpoint in checkpoint_specs:
            print(f"Running {label}: {checkpoint}", flush=True)
            model = load_darkir(checkpoint, device, base)
            for row, (_, _, _, image, _) in enumerate(selected):
                tensor = to_tensor(image)
                restored = restore(model, tensor, device)
                output = detector([restored])[0]
                condition_predictions[label][row] = [
                    {"box": box.tolist(), "category_id": int(class_id), "score": float(score)}
                    for box, class_id, score in zip(
                        output["boxes"].cpu(), output["labels"].cpu(), output["scores"].cpu()
                    )
                    if int(class_id) in mapped_ids and float(score) >= args.score_threshold
                ]
            del model
            torch.cuda.empty_cache()

    columns = ["Ground truth", *condition_predictions]
    margin, gap, panel_width, panel_height = 28, 12, 500, 300
    columns_y, row_start_y, row_label_height = 96, 134, 42
    canvas_width = margin * 2 + len(columns) * panel_width + (len(columns) - 1) * gap
    row_height = row_label_height + panel_height + 22
    canvas_height = row_start_y + len(selected) * row_height + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    draw.text((margin, 20), "ExDark V2 detection overlays (official validation)", fill="#17212b", font=font)
    draw.text((margin, 52), "Green: ground truth    Orange: detector predictions", fill="#34495e", font=small_font)

    for column, name in enumerate(columns):
        x = margin + column * (panel_width + gap)
        draw.text((x + 4, columns_y), name, fill="#17212b", font=label_font)

    for row, (image_id, image_path, class_name, image, ground_truth) in enumerate(selected):
        y = row_start_y + row * row_height
        draw.text(
            (margin, y),
            f"Example {row + 1}: {image_path.name} ({class_name})",
            fill="#273746",
            font=small_font,
        )
        for column, condition in enumerate(columns):
            x = margin + column * (panel_width + gap)
            panel = image.copy()
            if condition == "Ground truth":
                for boxes in ground_truth.values():
                    draw_detections(panel, [{"box": box} for box in boxes], "#19a66a")
            else:
                predictions = sorted(
                    condition_predictions[condition][row],
                    key=lambda prediction: float(prediction["score"]),
                    reverse=True,
                )[:5]
                draw_detections(panel, predictions, "#ee6c2f")
            fitted = ImageOps.contain(
                panel,
                (panel_width - 12, panel_height),
                method=Image.Resampling.LANCZOS,
            )
            px = x + (panel_width - fitted.width) // 2
            py = y + row_label_height + (panel_height - fitted.height) // 2
            canvas.paste(fitted, (px, py))
            draw.rectangle((px, py, px + fitted.width - 1, py + fitted.height - 1), outline="#d0d7de")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    print(f"Saved {len(selected)}-image comparison to {args.output}", flush=True)
    print(f"Device: {torch.cuda.get_device_name(device)}", flush=True)


if __name__ == "__main__":
    main()
