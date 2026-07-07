from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

# ============================================================
# Default configuration
# ============================================================

DEFAULT_SOURCE_DATASET = "datasource"
DEFAULT_DEST_DATASET = "datasource_4classes"

# Mapping format: old_class_id -> new_class_id
# Original classes in datasource/data.yaml:
# 0: bike, 1: helmet, 2: lisc, 3: no_helmet, 4: noise, 5: rider
CLASS_MAPPING = {
    0: 0,  # bike
    1: 1,  # helmet
    3: 2,  # no_helmet
    5: 3,  # rider
}

CLASS_NAMES = [
    "bike",
    "helmet",
    "no_helmet",
    "rider",
]

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================================================
# Helpers
# ============================================================


def find_image(image_folder: Path, stem: str) -> Path | None:
    """Find an image whose filename stem matches a YOLO label filename."""
    for ext in IMAGE_EXTENSIONS:
        candidates = [image_folder / f"{stem}{ext}", image_folder / f"{stem}{ext.upper()}"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def convert_label_lines(lines: Iterable[str]) -> list[str]:
    """Keep only mapped classes and rewrite old class ids to new class ids."""
    converted: list[str] = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            print(f"  [WARN] Skipping malformed label line {line_number}: {line}")
            continue

        try:
            old_class_id = int(parts[0])
        except ValueError:
            print(f"  [WARN] Skipping label line with invalid class id {line_number}: {line}")
            continue

        if old_class_id not in CLASS_MAPPING:
            continue

        parts[0] = str(CLASS_MAPPING[old_class_id])
        converted.append(" ".join(parts))

    return converted


def write_data_yaml(dest_dataset: Path) -> None:
    yaml_lines = [
        f"path: {dest_dataset.as_posix()}",
        "",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "",
        "names:",
    ]

    for class_id, class_name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {class_id}: {class_name}")

    (dest_dataset / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def prepare_destination(dest_dataset: Path, overwrite: bool) -> None:
    if dest_dataset.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dest_dataset}. "
                "Use --overwrite if you want to recreate it."
            )
        shutil.rmtree(dest_dataset)

    for split in SPLITS:
        (dest_dataset / split / "images").mkdir(parents=True, exist_ok=True)
        (dest_dataset / split / "labels").mkdir(parents=True, exist_ok=True)


def convert_dataset(source_dataset: Path, dest_dataset: Path, overwrite: bool, keep_empty: bool) -> None:
    if not source_dataset.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset}")

    prepare_destination(dest_dataset, overwrite=overwrite)

    totals = {
        "labels_read": 0,
        "images_written": 0,
        "labels_written": 0,
        "boxes_written": 0,
        "missing_images": 0,
        "empty_after_filter": 0,
    }

    for split in SPLITS:
        src_img = source_dataset / split / "images"
        src_lbl = source_dataset / split / "labels"
        dst_img = dest_dataset / split / "images"
        dst_lbl = dest_dataset / split / "labels"

        if not src_lbl.exists():
            print(f"\n[WARN] Missing label folder for split '{split}': {src_lbl}")
            continue

        label_files = sorted(src_lbl.glob("*.txt"))
        print(f"\nProcessing {split}: {len(label_files)} label files")

        split_images = 0
        split_labels = 0
        split_boxes = 0
        split_empty = 0

        for label_path in label_files:
            totals["labels_read"] += 1

            image_path = find_image(src_img, label_path.stem)
            if image_path is None:
                totals["missing_images"] += 1
                print(f"  [WARN] Image not found for label: {label_path.name}")
                continue

            new_lines = convert_label_lines(label_path.read_text(encoding="utf-8").splitlines())

            if not new_lines and not keep_empty:
                totals["empty_after_filter"] += 1
                split_empty += 1
                continue

            shutil.copy2(image_path, dst_img / image_path.name)
            (dst_lbl / label_path.name).write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

            split_images += 1
            split_labels += 1
            split_boxes += len(new_lines)

        totals["images_written"] += split_images
        totals["labels_written"] += split_labels
        totals["boxes_written"] += split_boxes

        print(f"  images written: {split_images}")
        print(f"  labels written: {split_labels}")
        print(f"  boxes written : {split_boxes}")
        print(f"  skipped empty : {split_empty}")

    write_data_yaml(dest_dataset)

    print("\nDone")
    print(f"Source     : {source_dataset}")
    print(f"Destination: {dest_dataset}")
    print(f"Labels read        : {totals['labels_read']}")
    print(f"Images written     : {totals['images_written']}")
    print(f"Labels written     : {totals['labels_written']}")
    print(f"Boxes written      : {totals['boxes_written']}")
    print(f"Missing images     : {totals['missing_images']}")
    print(f"Skipped empty files: {totals['empty_after_filter']}")
    print("Created data.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reduced YOLO dataset by keeping selected classes and remapping class ids."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE_DATASET, help="Source YOLO dataset folder")
    parser.add_argument("--dest", default=DEFAULT_DEST_DATASET, help="Destination YOLO dataset folder")
    parser.add_argument("--overwrite", action="store_true", help="Delete destination folder before recreating it")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep images/labels even when all boxes are removed after filtering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(
        source_dataset=Path(args.source),
        dest_dataset=Path(args.dest),
        overwrite=args.overwrite,
        keep_empty=args.keep_empty,
    )


if __name__ == "__main__":
    main()
