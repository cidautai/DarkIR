#!/usr/bin/env python3
# Team responsibility: Hemanth Sai -- matched DarkIR fine-tuning experiments.
"""Matched fine-tuning control/task-aware experiment for DarkIR on LOL-Blur."""
import argparse
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from archs.DarkIR import DarkIR


class LOLBlurPairs(Dataset):
    def __init__(self, root, crop_size=256):
        root = Path(root)
        low_root = root / "low_blur_noise"
        ref_root = root / "high_sharp_scaled"
        if not low_root.is_dir() or not ref_root.is_dir():
            raise FileNotFoundError(f"Expected {low_root} and {ref_root}")
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        low = {p.relative_to(low_root).as_posix(): p for p in low_root.rglob("*") if p.is_file() and p.suffix.lower() in exts}
        ref = {p.relative_to(ref_root).as_posix(): p for p in ref_root.rglob("*") if p.is_file() and p.suffix.lower() in exts}
        if not low or low.keys() != ref.keys():
            raise RuntimeError(f"Pair mismatch: inputs={len(low)}, references={len(ref)}, missing_ref={len(low.keys()-ref.keys())}, missing_input={len(ref.keys()-low.keys())}")
        self.pairs = [(low[k], ref[k]) for k in sorted(low)]
        self.crop_size = crop_size
        print(f"Verified {len(self.pairs)} paired training images")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, ref_path = self.pairs[idx]
        low = Image.open(low_path).convert("RGB")
        ref = Image.open(ref_path).convert("RGB")
        if low.size != ref.size:
            raise RuntimeError(f"Pair dimensions differ: {low_path} {low.size}, {ref_path} {ref.size}")
        w, h = low.size
        crop = min(self.crop_size, w, h)
        left = random.randint(0, w - crop)
        top = random.randint(0, h - crop)
        box = (left, top, left + crop, top + crop)
        low = low.crop(box)
        ref = ref.crop(box)
        low = torch.from_numpy(np.asarray(low).copy()).permute(2, 0, 1).float().div_(255)
        ref = torch.from_numpy(np.asarray(ref).copy()).permute(2, 0, 1).float().div_(255)
        if random.random() < 0.5:
            low, ref = low.flip(-1), ref.flip(-1)
        if random.random() < 0.5:
            low, ref = low.flip(-2), ref.flip(-2)
        return low, ref


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_darkir(checkpoint_path, device):
    model = DarkIR(
        img_channel=3, width=32,
        middle_blk_num_enc=2, middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9], extra_depth_wise=True,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt
    weights = {k.removeprefix("module."): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    return model.to(device)


def make_frozen_detector_backbone(device):
    detector = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        weights_backbone=None,
    )
    backbone = detector.backbone.to(device).eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    del detector
    return backbone


def detector_normalize(x, mean, std):
    return (x - mean) / std


def task_feature_loss(backbone, restored, reference, mean, std):
    # The detector stays frozen, but gradients still flow through it to restored pixels.
    with torch.no_grad():
        ref_features = backbone(detector_normalize(reference.clamp(0, 1), mean, std))
    out_features = backbone(detector_normalize(restored.clamp(0, 1), mean, std))
    losses = []
    for key in ref_features:
        a = F.normalize(out_features[key], p=2, dim=1, eps=1e-6)
        b = F.normalize(ref_features[key], p=2, dim=1, eps=1e-6)
        losses.append((1.0 - (a * b).sum(dim=1)).mean())
    return torch.stack(losses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["control", "task"], required=True)
    ap.add_argument("--root", default="data/datasets/LOLBlur/train")
    ap.add_argument("--checkpoint", default="models/DarkIR_384.pt")
    ap.add_argument("--output", default=None)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--task-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        ap.error("--steps and --batch-size must be positive")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available; refusing to run a slow CPU training pilot")
    print(f"Mode: {args.mode} | Device: {torch.cuda.get_device_name(0)}")

    model = load_darkir(args.checkpoint, device)
    backbone = make_frozen_detector_backbone(device) if args.mode == "task" else None
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Re-seed after model construction so control/task runs use a matched sample order.
    seed_all(args.seed)
    dataset = LOLBlurPairs(args.root, args.crop_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, drop_last=True,
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    torch.backends.cudnn.benchmark = True
    output_path = Path(args.output or f"models/DarkIR_{args.mode}_ft.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_all = time.time()
    for step in range(1, args.steps + 1):
        try:
            low, ref = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            low, ref = next(iterator)
        low = low.to(device, non_blocking=True)
        ref = ref.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        restored = model(low)
        loss_restore = F.l1_loss(restored, ref)
        if args.mode == "task":
            loss_task = task_feature_loss(backbone, restored, ref, mean, std)
            loss = loss_restore + args.task_weight * loss_task
        else:
            loss_task = restored.new_zeros(())
            loss = loss_restore
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % 5 == 0 or step == args.steps:
            torch.cuda.synchronize()
            elapsed = time.time() - start_all
            print(
                f"step={step}/{args.steps} total={loss.item():.6f} "
                f"restoration={loss_restore.item():.6f} task={loss_task.item():.6f} "
                f"grad_norm={float(grad_norm):.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    torch.save({
        "params": model.state_dict(),
        "mode": args.mode,
        "steps_completed": args.steps,
        "lr": args.lr,
        "task_weight": args.task_weight if args.mode == "task" else 0.0,
        "seed": args.seed,
    }, output_path)
    print(f"Saved checkpoint: {output_path}")
    print(f"Average time/step: {(time.time() - start_all) / args.steps:.3f}s")
    print("TRAINING PILOT PASSED")


if __name__ == "__main__":
    main()
