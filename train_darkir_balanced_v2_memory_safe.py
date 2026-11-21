#!/usr/bin/env python3
# Team responsibility: Hemanth Sai -- memory-safe Balanced V2 training implementation.
"""Balanced V2 DarkIR fine-tuning: restoration control or detector-guided.

Run from the DarkIR repository root. This local experiment script does not
modify repository source files or upload anything.

Each optimizer step uses a paired LOL-Blur batch for restoration supervision.
In detection mode, it additionally uses one disjoint ExDark detection-supervision
image and its real boxes through a frozen, ExDark-trained Faster R-CNN teacher.
The teacher is in train mode only because torchvision returns its loss_dict in
train mode; its parameters are frozen and ordinary BatchNorm statistics are held.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
REQUIRED_DETECTION_LOSSES = {
    "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


class LOLBlurManifestDataset(Dataset):
    """Paired LOL-Blur training data restricted to the scene-disjoint manifest."""

    def __init__(self, manifest_path: Path, root: Path, crop_size: int):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("dataset") != "LOLBlur" or payload.get("format_version") != 1:
            raise ValueError(f"Unsupported LOL-Blur manifest: {manifest_path}")
        if payload.get("split") != "train":
            raise ValueError("Restoration training requires lolblur_train.json, not validation")
        self.pairs = payload.get("pairs", [])
        self.root = root
        self.crop_size = crop_size
        if not self.pairs:
            raise ValueError(f"No paired images in {manifest_path}")
        if int(payload.get("num_pairs", -1)) != len(self.pairs):
            raise ValueError("LOL-Blur manifest num_pairs does not match pairs list")
        scenes = {item.get("scene") for item in self.pairs}
        if len(scenes) != int(payload.get("num_scenes", -1)):
            raise ValueError("LOL-Blur manifest scene count is inconsistent")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        record = self.pairs[index]
        low_path = self.root / record["low"]
        ref_path = self.root / record["reference"]
        if not low_path.is_file() or not ref_path.is_file():
            raise FileNotFoundError(f"Missing LOL-Blur pair: {low_path} | {ref_path}")
        with Image.open(low_path) as im:
            low = pil_to_tensor(im)
        with Image.open(ref_path) as im:
            reference = pil_to_tensor(im)
        if low.shape != reference.shape:
            raise RuntimeError(
                f"LOL-Blur pair shape mismatch: {record['low']} {tuple(low.shape)} vs "
                f"{record['reference']} {tuple(reference.shape)}"
            )

        _, height, width = low.shape
        pad_h = max(0, self.crop_size - height)
        pad_w = max(0, self.crop_size - width)
        if pad_h or pad_w:
            low = F.pad(low.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)
            reference = F.pad(
                reference.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate"
            ).squeeze(0)
            _, height, width = low.shape

        top = random.randint(0, height - self.crop_size)
        left = random.randint(0, width - self.crop_size)
        low = low[:, top:top + self.crop_size, left:left + self.crop_size]
        reference = reference[:, top:top + self.crop_size, left:left + self.crop_size]
        if random.random() < 0.5:
            low, reference = low.flip(-1), reference.flip(-1)
        if random.random() < 0.5:
            low, reference = low.flip(-2), reference.flip(-2)
        return low.contiguous(), reference.contiguous()


class ExDarkManifestDataset(Dataset):
    """Native-size ExDark images with real boxes from the reserved 600-image split."""

    def __init__(self, manifest_path: Path, root: Path, max_side: int):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("dataset") != "ExDark" or payload.get("format_version") != 1:
            raise ValueError(f"Unsupported ExDark manifest: {manifest_path}")
        if payload.get("split") != "detection_supervision":
            raise ValueError("Detection training requires exdark_detection_supervision.json")
        if payload.get("classes") != EXDARK_CLASSES or payload.get("label_ids") != list(range(1, 13)):
            raise ValueError("ExDark class ordering/IDs differ from the trained teacher")
        self.records = payload.get("records", [])
        self.root = root
        self.max_side = max_side
        if not self.records:
            raise ValueError("Detection-supervision manifest is empty")
        if int(payload.get("num_images", -1)) != len(self.records):
            raise ValueError("ExDark manifest num_images does not match records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = self.root / record["image"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            image = im.convert("RGB")
            orig_w, orig_h = image.size
            boxes = torch.tensor([row[:4] for row in record["boxes"]], dtype=torch.float32).reshape(-1, 4)
            labels = torch.tensor([int(row[4]) for row in record["boxes"]], dtype=torch.int64)
            if labels.numel() == 0 or torch.any(labels < 1) or torch.any(labels > 12):
                raise RuntimeError(f"Invalid/missing class labels in {path}")
            if torch.any(boxes[:, 2:] <= boxes[:, :2]):
                raise RuntimeError(f"Degenerate bounding box in {path}")
            if (
                torch.any(boxes[:, 0] < 0) or torch.any(boxes[:, 1] < 0)
                or torch.any(boxes[:, 2] > orig_w + 1e-3)
                or torch.any(boxes[:, 3] > orig_h + 1e-3)
            ):
                raise RuntimeError(f"Bounding box outside source image in {path}")

            if self.max_side > 0 and max(orig_w, orig_h) > self.max_side:
                scale = self.max_side / max(orig_w, orig_h)
                new_w = max(1, round(orig_w * scale))
                new_h = max(1, round(orig_h * scale))
                image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
                boxes[:, [0, 2]] *= new_w / orig_w
                boxes[:, [1, 3]] *= new_h / orig_h

        image_tensor = pil_to_tensor(image)
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


class SobelEdgeLoss(nn.Module):
    """Device-agnostic RGB Sobel gradient L1 loss."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 8.0
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _gradients(self, image: torch.Tensor):
        channels = image.shape[1]
        kx = self.kx.to(dtype=image.dtype).expand(channels, 1, 3, 3)
        ky = self.ky.to(dtype=image.dtype).expand(channels, 1, 3, 3)
        padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(padded, kx, groups=channels)
        gy = F.conv2d(padded, ky, groups=channels)
        return gx, gy

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_x, pred_y = self._gradients(prediction)
        target_x, target_y = self._gradients(target)
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


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
    return model.to(device)


def load_frozen_teacher(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("class_names") != EXDARK_CLASSES:
        raise ValueError("Teacher checkpoint class order is not the expected ExDark order")
    if checkpoint.get("num_classes_with_background") != 13:
        raise ValueError("Expected a 13-class-output (background + 12 classes) teacher")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Teacher checkpoint is missing model_state_dict")

    teacher = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = teacher.roi_heads.box_predictor.cls_score.in_features
    teacher.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=13)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.roi_heads.score_thresh = 0.001
    teacher.roi_heads.detections_per_img = 100
    teacher.to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    # torchvision Faster R-CNN must be in train mode to return its four loss terms.
    teacher.train()
    for module in teacher.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return teacher


def next_batch(loader: DataLoader, iterator, name: str):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, args,
                    step: int, losses: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": model.state_dict(),
        "experiment": "Balanced V2 detector-guided low-light restoration",
        "mode": args.mode,
        "steps_completed": step,
        "total_steps_requested": args.steps,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss_weights": {
            "lpips": args.lpips_weight,
            "edge": args.edge_weight,
            "detection": args.detection_weight if args.mode == "detection" else 0.0,
        },
        "teacher_checkpoint": str(args.teacher_checkpoint) if args.mode == "detection" else None,
        "lolblur_manifest": str(args.lolblur_manifest),
        "exdark_manifest": str(args.exdark_manifest) if args.mode == "detection" else None,
        "seed": args.seed,
        "latest_losses": losses,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["control", "detection"], required=True)
    p.add_argument("--lolblur-root", type=Path, default=Path("data/datasets/LOLBlur/train"))
    p.add_argument("--lolblur-manifest", type=Path, default=Path("data/manifests/balanced_v2/lolblur_train.json"))
    p.add_argument("--exdark-root", type=Path, default=Path("data/datasets/ExDark"))
    p.add_argument("--exdark-manifest", type=Path, default=Path("data/manifests/balanced_v2/exdark_detection_supervision.json"))
    p.add_argument("--darkir-checkpoint", type=Path, default=Path("models/DarkIR_384.pt"))
    p.add_argument("--teacher-checkpoint", type=Path, default=Path("models/exdark_teacher_v2/best.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("models/balanced_v2_pilots"))
    p.add_argument("--steps", type=int, default=200, help="Pilot steps; use a longer schedule only after validation")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--det-max-side", type=int, default=640, help="Resize ExDark inputs before DarkIR; 640 is the safe pilot default; 0 disables")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lpips-weight", type=float, default=0.10)
    p.add_argument("--edge-weight", type=float, default=0.05)
    p.add_argument("--detection-weight", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--no-save", action="store_true", help="Do not write any checkpoint")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.workers < 0 or args.log_every < 1 or args.save_every < 1:
        raise ValueError("steps, batch-size, log-every, save-every must be positive; workers nonnegative")
    if args.lr <= 0 or min(args.lpips_weight, args.edge_weight, args.detection_weight) < 0:
        raise ValueError("Learning rate must be positive and all loss weights nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run this training pilot on CPU")
    if args.mode == "detection" and not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(args.teacher_checkpoint)

    seed_everything(args.seed)
    device = torch.device("cuda:0")
    print(f"Mode: {args.mode} | Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Torch: {torch.__version__} | torchvision: {__import__('torchvision').__version__}", flush=True)

    lol_ds = LOLBlurManifestDataset(args.lolblur_manifest, args.lolblur_root, args.crop_size)
    lol_generator = torch.Generator().manual_seed(args.seed)
    lol_loader = DataLoader(
        lol_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker, generator=lol_generator, drop_last=True,
    )
    if len(lol_loader) == 0:
        raise RuntimeError("LOL-Blur training loader has zero batches")
    print(f"LOL-Blur restoration data: {len(lol_ds)} pairs | {len(lol_loader)} batches/epoch", flush=True)

    exdark_loader = None
    exdark_iterator = None
    teacher = None
    if args.mode == "detection":
        exdark_ds = ExDarkManifestDataset(args.exdark_manifest, args.exdark_root, args.det_max_side)
        exdark_generator = torch.Generator().manual_seed(args.seed + 1)
        exdark_loader = DataLoader(
            exdark_ds, batch_size=1, shuffle=True, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
            worker_init_fn=seed_worker, generator=exdark_generator,
            collate_fn=detection_collate, drop_last=True,
        )
        if len(exdark_loader) == 0:
            raise RuntimeError("ExDark detection-supervision loader has zero batches")
        print(
            f"ExDark detection supervision: {len(exdark_ds)} images | "
            f"{len(exdark_loader)} batches/epoch | max_side={args.det_max_side}", flush=True
        )

    darkir = load_darkir(args.darkir_checkpoint, device)
    darkir.train()
    print(f"Loaded DarkIR: {sum(p.numel() for p in darkir.parameters()):,} parameters", flush=True)

    if args.mode == "detection":
        teacher = load_frozen_teacher(args.teacher_checkpoint, device)
        trainable = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
        if trainable != 0:
            raise RuntimeError(f"Teacher is not frozen: {trainable} parameters trainable")
        print("Loaded frozen 12-class ExDark teacher.", flush=True)

    # LPIPS is a perceptual metric/network; keep it frozen but differentiable wrt input.
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS package is required. Install the repo requirement: pip install lpips==0.1.4") from exc
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    edge_loss_fn = SobelEdgeLoss().to(device)

    optimizer = torch.optim.AdamW(darkir.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )
    lol_iterator = iter(lol_loader)
    if exdark_loader is not None:
        exdark_iterator = iter(exdark_loader)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"DarkIR_v2_{args.mode}_pilot_{args.steps}.pt"
    if not args.no_save:
        expected_outputs = [output_path]
        expected_outputs.extend(
            output_dir / f"DarkIR_v2_{args.mode}_step{step:06d}.pt"
            for step in range(args.save_every, args.steps, args.save_every)
        )
        existing = [path for path in expected_outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing pilot checkpoint(s): "
                + ", ".join(str(path) for path in existing)
                + ". Choose another --output-dir or move the old pilot files."
            )

    running: dict[str, float] = {key: 0.0 for key in ["total", "l1", "lpips", "edge", "detection"]}
    start = time.time()
    torch.backends.cudnn.benchmark = True
    for step in range(1, args.steps + 1):
        (low, reference), lol_iterator = next_batch(lol_loader, lol_iterator, "LOL-Blur")
        low = low.to(device, non_blocking=True)
        reference = reference.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Backpropagate and release the LOL-Blur graph before building the much
        # larger full-resolution ExDark + detector graph. This is mathematically
        # equivalent to backpropagating the sum because both branches share only
        # DarkIR parameters and gradients accumulate until optimizer.step().
        restored_pairs = darkir(low)
        if restored_pairs.shape != reference.shape:
            raise RuntimeError(
                f"DarkIR paired output/reference shape mismatch: {tuple(restored_pairs.shape)} vs {tuple(reference.shape)}"
            )
        l1_value = F.l1_loss(restored_pairs, reference)
        lpips_value = perceptual(
            restored_pairs.clamp(0.0, 1.0) * 2.0 - 1.0,
            reference * 2.0 - 1.0,
        ).mean()
        edge_value = edge_loss_fn(restored_pairs, reference)
        restoration_loss = l1_value + args.lpips_weight * lpips_value + args.edge_weight * edge_value
        if not torch.isfinite(restoration_loss):
            raise RuntimeError(f"Non-finite restoration loss at step {step}")

        l1_scalar = float(l1_value.detach().float().cpu())
        lpips_scalar = float(lpips_value.detach().float().cpu())
        edge_scalar = float(edge_value.detach().float().cpu())
        restoration_scalar = float(restoration_loss.detach().float().cpu())
        restoration_loss.backward()
        del restored_pairs, l1_value, lpips_value, edge_value, restoration_loss

        detection_scalar = 0.0
        loss_parts: dict[str, float] = {}
        if args.mode == "detection":
            (det_images, det_targets), exdark_iterator = next_batch(
                exdark_loader, exdark_iterator, "ExDark"
            )
            det_image = det_images[0].to(device, non_blocking=True)
            target = {
                key: value.to(device, non_blocking=True)
                for key, value in det_targets[0].items()
            }
            restored_detection = darkir(det_image.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)
            detector_losses = teacher([restored_detection], [target])
            missing = REQUIRED_DETECTION_LOSSES - set(detector_losses)
            if missing:
                raise RuntimeError(f"Teacher did not return required detection losses: {sorted(missing)}")
            for key, value in detector_losses.items():
                if not torch.isfinite(value):
                    raise RuntimeError(f"Non-finite {key} at step {step}: {float(value.detach().cpu())}")
                loss_parts[key] = float(value.detach().float().cpu())
            detection_value = sum(detector_losses.values())
            if not torch.isfinite(detection_value):
                raise RuntimeError(f"Non-finite detection loss at step {step}")
            detection_scalar = float(detection_value.detach().float().cpu())
            (args.detection_weight * detection_value).backward()
            del restored_detection, detector_losses, detection_value, det_image, det_images, det_targets, target

        total_scalar = restoration_scalar + args.detection_weight * detection_scalar
        if not np.isfinite(total_scalar):
            raise RuntimeError(f"Non-finite total loss at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(darkir.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm) or float(grad_norm) <= 0.0:
            raise RuntimeError(f"Invalid DarkIR gradient norm at step {step}: {float(grad_norm)}")
        if args.mode == "detection":
            teacher_grad_tensors = sum(p.grad is not None for p in teacher.parameters())
            if teacher_grad_tensors:
                raise RuntimeError(f"Frozen teacher unexpectedly has {teacher_grad_tensors} parameter gradients")
        optimizer.step()
        scheduler.step()

        values = {
            "total": total_scalar,
            "l1": l1_scalar,
            "lpips": lpips_scalar,
            "edge": edge_scalar,
            "detection": detection_scalar,
        }
        for key, value in values.items():
            running[key] += value

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            print(
                f"step={step}/{args.steps} total={values['total']:.6f} "
                f"L1={values['l1']:.6f} LPIPS={values['lpips']:.6f} "
                f"edge={values['edge']:.6f} detection={values['detection']:.6f} "
                f"det_weighted={args.detection_weight * values['detection']:.6f} "
                f"grad_norm_preclip={float(grad_norm):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if loss_parts:
                print("  detection components: " + ", ".join(
                    f"{k}={v:.6f}" for k, v in sorted(loss_parts.items())
                ), flush=True)

        if not args.no_save and (step % args.save_every == 0 or step == args.steps):
            if step == args.steps:
                save_path = output_path
            else:
                save_path = output_dir / f"DarkIR_v2_{args.mode}_step{step:06d}.pt"
            save_checkpoint(save_path, darkir, optimizer, scheduler, args, step, values)
            print(f"Saved checkpoint: {save_path}", flush=True)

    averages = {key: value / args.steps for key, value in running.items()}
    print("TRAINING PILOT COMPLETED", flush=True)
    print("Mean losses: " + json.dumps(averages, indent=2), flush=True)
    print(f"Average seconds/step: {(time.time() - start) / args.steps:.3f}", flush=True)
    if args.mode == "detection":
        print("Teacher parameter gradients: 0 (verified throughout training)", flush=True)
    if args.no_save:
        print("--no-save set: no checkpoint was written", flush=True)
    else:
        print(f"Pilot checkpoint: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
