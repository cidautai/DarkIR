"""
Team responsibility: Bhanu Prakash -- Balanced V2 dataset and partition audit.

This script verifies LOLBlur scene/pair separation and ExDark official split
resolution without modifying datasets or checkpoints.
"""
from pathlib import Path
from collections import Counter
from PIL import Image

from evaluate_exdark_cocoeval import (
    EXDARK_CLASSES,
    EXDARK_TO_COCO_NAME,
    parse_split_file,
    build_image_index,
    build_annotation_index,
    resolve_image,
    resolve_annotation,
    parse_gt_file,
)

ROOT = Path("data/datasets")
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_map(root):
    return {
        p.relative_to(root).as_posix(): p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTS
        and "__MACOSX" not in p.parts
        and not p.name.startswith("._")
    }


def audit_lolblur():
    print("\n========== LOL-BLUR AUDIT ==========")

    split_data = {}

    for split in ["train", "test"]:
        root = ROOT / "LOLBlur" / split
        low_root = root / "low_blur_noise"
        gt_root = root / "high_sharp_scaled"

        low = image_map(low_root)
        gt = image_map(gt_root)

        common = set(low) & set(gt)

        missing_gt = set(low) - set(gt)
        missing_low = set(gt) - set(low)

        dimension_errors = []
        read_errors = []

        for key in sorted(common):
            try:
                with Image.open(low[key]) as a:
                    low_size = a.size
                with Image.open(gt[key]) as b:
                    gt_size = b.size

                if low_size != gt_size:
                    dimension_errors.append(key)

            except Exception as exc:
                read_errors.append((key, str(exc)))

        groups = {Path(k).parts[0] for k in common}

        split_data[split] = {
            "pairs": common,
            "groups": groups,
        }

        print(f"\n{split.upper()}")
        print("Low-light images:", len(low))
        print("Reference images:", len(gt))
        print("Matching pairs:", len(common))
        print("Missing references:", len(missing_gt))
        print("Missing inputs:", len(missing_low))
        print("Dimension mismatches:", len(dimension_errors))
        print("Unreadable pairs:", len(read_errors))
        print("Scene folders:", len(groups))

        if dimension_errors:
            print("Dimension mismatch samples:", dimension_errors[:5])

        if read_errors:
            print("Unreadable samples:", read_errors[:3])

    overlap = (
        split_data["train"]["pairs"]
        & split_data["test"]["pairs"]
    )

    shared_groups = (
        split_data["train"]["groups"]
        & split_data["test"]["groups"]
    )

    print("\nLOL-Blur train/test exact pair overlap:", len(overlap))
    print("LOL-Blur shared scene-folder IDs:", len(shared_groups))


def audit_exdark():
    print("\n========== EXDARK AUDIT ==========")

    split_file = ROOT / "ExDark/Groundtruth/imageclasslist.txt"
    images_root = ROOT / "ExDark/images/ExDark"
    annotations_root = ROOT / "ExDark/Groundtruth/ExDark_Annno"

    image_index = build_image_index(images_root)
    annotation_index = build_annotation_index(annotations_root)

    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    coco_categories = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

    coco_ids = {
        name.lower(): coco_categories.index(EXDARK_TO_COCO_NAME[name.lower()])
        for name in EXDARK_CLASSES
    }

    resolved_by_split = {}
    expected = {1: 3000, 2: 1800, 3: 2563}

    for code, label in [(1, "TRAIN"), (2, "VALIDATION"), (3, "TEST")]:
        records = parse_split_file(split_file, code)

        image_counts = Counter()
        box_counts = Counter()
        resolved_images = set()

        missing_images = []
        missing_annotations = []
        parse_errors = []

        total_boxes = 0

        for rel, class_name in records:
            image_counts[class_name] += 1

            try:
                image_path = resolve_image(
                    rel, class_name, images_root, image_index
                )
                resolved_images.add(str(image_path.resolve()))

                annotation_path = resolve_annotation(
                    image_path,
                    class_name,
                    annotations_root,
                    annotation_index,
                )

                with Image.open(image_path) as im:
                    width, height = im.size

                boxes = parse_gt_file(
                    annotation_path,
                    width,
                    height,
                    coco_ids,
                )

                for class_id, values in boxes.items():
                    name = next(
                        k for k, v in coco_ids.items() if v == class_id
                    )
                    box_counts[name] += len(values)
                    total_boxes += len(values)

            except FileNotFoundError as exc:
                message = str(exc)
                if "Annotation" in message:
                    missing_annotations.append(rel)
                else:
                    missing_images.append(rel)

            except Exception as exc:
                parse_errors.append((rel, str(exc)))

        resolved_by_split[code] = resolved_images

        print(f"\n{label}")
        print("Images in official split:", len(records))
        print("Expected images:", expected[code])
        print("Primary image-class distribution:", dict(image_counts))
        print("Parsed bounding boxes:", total_boxes)
        print("Bounding boxes by class:", dict(box_counts))
        print("Missing images:", len(missing_images))
        print("Missing annotations:", len(missing_annotations))
        print("Parsing errors:", len(parse_errors))

        if missing_images:
            print("Missing image samples:", missing_images[:5])

        if missing_annotations:
            print("Missing annotation samples:", missing_annotations[:5])

        if parse_errors:
            print("Parsing error samples:", parse_errors[:3])

    for a, b in [(1, 2), (1, 3), (2, 3)]:
        overlap = resolved_by_split[a] & resolved_by_split[b]
        print(f"\nSplit overlap ({a}, {b}):", len(overlap))


if __name__ == "__main__":
    audit_lolblur()
    audit_exdark()
    print("\nDATA AUDIT FINISHED")
