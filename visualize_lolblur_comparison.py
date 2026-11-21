#!/usr/bin/env python3
# Team responsibility: Hemanth Sai -- LOLBlur qualitative comparison.
from pathlib import Path
import re
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms.functional import to_tensor, to_pil_image
from archs.DarkIR import DarkIR

ROOT = Path("data/datasets/LOLBlur/test")
LOW_DIR = ROOT / "low_blur_noise"
GT_DIR = ROOT / "high_sharp_scaled"

OUTPUT = Path("results/qualitative_lolblur_2026-11-10")
OUTPUT.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "Pretrained": "models/DarkIR_384.pt",
    "Control 50k": "models/DarkIR_control_transformmatched_50000.pt",
    "Task-aware 50k": "models/DarkIR_task_transformmatched_50000.pt",
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def natural_key(path):
    relative_path = path.relative_to(LOW_DIR).as_posix()
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", relative_path)
    ]


def load_model(checkpoint_path):
    model = DarkIR(
        img_channel=3,
        width=32,
        middle_blk_num_enc=2,
        middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3],
        dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9],
        extra_depth_wise=True,
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    weights = checkpoint["params"] if "params" in checkpoint else checkpoint
    weights = {
        k.removeprefix("module."): v for k, v in weights.items()
    }

    model.load_state_dict(weights, strict=True)
    return model.to(DEVICE).eval()


# Verify dataset folders and checkpoint files.
if not LOW_DIR.is_dir() or not GT_DIR.is_dir():
    raise FileNotFoundError(f"Dataset folders not found under {ROOT}")

for name, path in CHECKPOINTS.items():
    if not Path(path).is_file():
        raise FileNotFoundError(f"{name} checkpoint not found: {path}")

# Match input and reference images by filename.
low_files = sorted(
    [
        p for p in LOW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ],
    key=lambda p: p.relative_to(LOW_DIR).as_posix().lower(),
)

pairs = [
    (p, GT_DIR / p.relative_to(LOW_DIR))
    for p in low_files
    if (GT_DIR / p.relative_to(LOW_DIR)).is_file()
]

if not pairs:
    raise RuntimeError("No matching low-light / ground-truth image pairs found.")

# Choose six evenly spaced, reproducible samples.
count = min(6, len(pairs))
indices = np.linspace(0, len(pairs) - 1, count, dtype=int)
selected = [pairs[i] for i in indices]

print(f"Device: {DEVICE}")
print(f"Matched test pairs: {len(pairs)}")
print("Selected images:")
for low, gt in selected:
    print(f"  {low.name}")

# Save the selected input and ground-truth images.
images = {}
for low, gt in selected:
    key = low.relative_to(LOW_DIR).with_suffix("").as_posix().replace("/", "_")
    images[key] = {
        "Input": Image.open(low).convert("RGB"),
        "Ground truth": Image.open(gt).convert("RGB"),
    }

    images_dir = OUTPUT / "images" / key
    images_dir.mkdir(parents=True, exist_ok=True)
    images[key]["Input"].save(images_dir / "input.png")
    images[key]["Ground truth"].save(images_dir / "ground_truth.png")

# Run all three models on exactly the same inputs.
for model_name, checkpoint_path in CHECKPOINTS.items():
    print(f"\nLoading {model_name}: {checkpoint_path}")
    model = load_model(checkpoint_path)

    for low, _ in selected:
        key = low.relative_to(LOW_DIR).with_suffix("").as_posix().replace("/", "_")
        image = images[key]["Input"]
        x = to_tensor(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            output = model(x).clamp(0, 1)

        restored = to_pil_image(output[0].cpu())

        images_dir = OUTPUT / "images" / key
        safe_name = model_name.lower().replace(" ", "_")
        restored.save(images_dir / f"{safe_name}.png")
        images[key][model_name] = restored

        print(f"  Restored: {low.relative_to(LOW_DIR)}")

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# Build a five-column comparison grid.
columns = ["Input", "Ground truth", "Pretrained", "Control 50k", "Task-aware 50k"]

margin = 25
column_width = 360
image_height = 250
header_height = 35
row_gap = 25
row_height = header_height + image_height + row_gap

canvas_width = margin * 2 + len(columns) * column_width
canvas_height = margin * 2 + 45 + count * row_height

canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default()

draw.text(
    (margin, margin),
    "LOL-Blur: qualitative restoration comparison",
    fill="black",
    font=font,
)

for col, column_name in enumerate(columns):
    x = margin + col * column_width

    draw.text(
        (x + 8, margin + 45),
        column_name,
        fill="black",
        font=font,
    )

for row, (low, _) in enumerate(selected):
    key = low.relative_to(LOW_DIR).with_suffix("").as_posix().replace("/", "_")
    y = margin + 45 + row * row_height + header_height

    draw.text(
        (margin, y - 18),
        f"Sample {row + 1}: {low.relative_to(LOW_DIR)}",
        fill="black",
        font=font,
    )

    for col, column_name in enumerate(columns):
        x = margin + col * column_width
        image = images[key][column_name]

        fitted = ImageOps.contain(
            image,
            (column_width - 16, image_height),
            method=Image.Resampling.LANCZOS,
        )

        paste_x = x + (column_width - fitted.width) // 2
        paste_y = y + (image_height - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))

grid_path = OUTPUT / "comparison_grid.png"
canvas.save(grid_path)

with open(OUTPUT / "selected_samples.txt", "w") as f:
    for low, _ in selected:
        f.write(low.relative_to(LOW_DIR).as_posix() + "\n")

print("\nQUALITATIVE COMPARISON COMPLETED")
print(f"Grid saved to: {grid_path}")
print(f"Individual images saved under: {OUTPUT / 'images'}")
