from pathlib import Path
from PIL import Image
import hashlib
from collections import Counter, defaultdict

# ============================================================
# RoadVision - Dataset Integrity Audit
# READ-ONLY: This script does NOT modify any dataset files.
# ============================================================

ROOT = Path(r"C:\Users\VICTUS\OneDrive\Documents\RoadVision\dataset")

SPLITS = {
    "train": {
        "images": ROOT / "images" / "train",
        "labels": ROOT / "labels" / "train",
    },
    "val": {
        "images": ROOT / "images" / "val",
        "labels": ROOT / "labels" / "val",
    },
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_CLASS_IDS = {0, 1, 2, 3}

print("=" * 72)
print("RoadVision - Dataset Integrity Audit")
print("=" * 72)
print()
print("READ-ONLY AUDIT: No files will be changed.")
print()

overall_problems = []
split_stats = {}
image_hashes = defaultdict(list)


def image_key(path):
    return path.stem.lower()


def audit_split(split_name, image_dir, label_dir):
    print("=" * 72)
    print(f"{split_name.upper()} DATASET")
    print("=" * 72)

    problems = []

    if not image_dir.exists():
        print(f"ERROR: Image directory does not exist: {image_dir}")
        return

    if not label_dir.exists():
        print(f"ERROR: Label directory does not exist: {label_dir}")
        return

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    labels = [
        p for p in label_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ]

    image_by_stem = defaultdict(list)
    label_by_stem = defaultdict(list)

    for p in images:
        image_by_stem[image_key(p)].append(p)

    for p in labels:
        label_by_stem[image_key(p)].append(p)

    duplicate_image_names = {
        k: v for k, v in image_by_stem.items()
        if len(v) > 1
    }

    duplicate_label_names = {
        k: v for k, v in label_by_stem.items()
        if len(v) > 1
    }

    missing_labels = [
        p for p in images
        if image_key(p) not in label_by_stem
    ]

    orphan_labels = [
        p for p in labels
        if image_key(p) not in image_by_stem
    ]

    print(f"Images found          : {len(images)}")
    print(f"Labels found          : {len(labels)}")
    print(f"Missing labels        : {len(missing_labels)}")
    print(f"Orphan labels         : {len(orphan_labels)}")
    print(f"Duplicate image names : {len(duplicate_image_names)}")
    print(f"Duplicate label names : {len(duplicate_label_names)}")
    print()

    # --------------------------------------------------------
    # Image/label pairing problems
    # --------------------------------------------------------

    for stem, paths in duplicate_image_names.items():
        problems.append(
            f"{split_name}: duplicate image stem '{stem}': "
            + ", ".join(p.name for p in paths)
        )

    for stem, paths in duplicate_label_names.items():
        problems.append(
            f"{split_name}: duplicate label stem '{stem}': "
            + ", ".join(p.name for p in paths)
        )

    for p in missing_labels:
        problems.append(
            f"{split_name}: missing label for image: {p.name}"
        )

    for p in orphan_labels:
        problems.append(
            f"{split_name}: orphan label: {p.name}"
        )

    # --------------------------------------------------------
    # Check image corruption + calculate hashes
    # --------------------------------------------------------

    unreadable_images = []

    print("Checking image files...")

    for p in images:
        try:
            with Image.open(p) as im:
                im.verify()

            # Reopen after verify to make sure pixels can actually load.
            with Image.open(p) as im:
                im.load()

                width, height = im.size

                if width <= 0 or height <= 0:
                    raise ValueError("invalid image dimensions")

            file_hash = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()

            image_hashes[file_hash].append(
                (split_name, p)
            )

        except Exception as e:
            unreadable_images.append((p, str(e)))

            problems.append(
                f"{split_name}: unreadable/corrupt image: "
                f"{p.name} ({e})"
            )

    print(f"Unreadable/corrupt images : {len(unreadable_images)}")
    print()

    # --------------------------------------------------------
    # Check YOLO annotations
    # --------------------------------------------------------

    malformed_labels = []
    invalid_class_ids = []
    invalid_coordinates = []
    empty_labels = []

    class_counts = Counter()

    transverse_images = 0
    transverse_instances = 0

    print("Checking YOLO annotations...")

    for label in labels:

        try:
            raw = label.read_text(
                encoding="utf-8-sig"
            )

        except Exception as e:

            malformed_labels.append(
                (label, f"cannot read file: {e}")
            )

            problems.append(
                f"{split_name}: cannot read label "
                f"{label.name} ({e})"
            )

            continue

        lines = raw.splitlines()

        # Empty annotation files are reported but not automatically
        # treated as corruption.
        if not any(line.strip() for line in lines):
            empty_labels.append(label)
            continue

        transverse_here = 0

        for line_no, line in enumerate(lines, 1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # YOLO format:
            # class x_center y_center width height
            if len(parts) != 5:

                malformed_labels.append(
                    (
                        label,
                        f"line {line_no}: expected 5 values, "
                        f"got {len(parts)}"
                    )
                )

                problems.append(
                    f"{split_name}: malformed label "
                    f"{label.name}, line {line_no}"
                )

                continue

            try:
                class_id = int(parts[0])

                coords = [
                    float(x)
                    for x in parts[1:]
                ]

            except ValueError:

                malformed_labels.append(
                    (
                        label,
                        f"line {line_no}: non-numeric value"
                    )
                )

                problems.append(
                    f"{split_name}: non-numeric annotation "
                    f"in {label.name}, line {line_no}"
                )

                continue

            class_counts[class_id] += 1

            # Valid RoadVision classes:
            # 0 longitudinal
            # 1 transverse
            # 2 alligator
            # 3 pothole

            if class_id not in VALID_CLASS_IDS:

                invalid_class_ids.append(
                    (label, line_no, class_id)
                )

                problems.append(
                    f"{split_name}: invalid class ID "
                    f"{class_id} in {label.name}, "
                    f"line {line_no}"
                )

            # YOLO normalized coordinates should be 0..1.
            if any(x < 0 or x > 1 for x in coords):

                invalid_coordinates.append(
                    (label, line_no, coords)
                )

                problems.append(
                    f"{split_name}: invalid bbox coordinates "
                    f"in {label.name}, line {line_no}: {coords}"
                )

            # Width and height must be positive.
            if coords[2] <= 0 or coords[3] <= 0:

                invalid_coordinates.append(
                    (label, line_no, coords)
                )

                problems.append(
                    f"{split_name}: zero/negative bbox size "
                    f"in {label.name}, line {line_no}"
                )

            if class_id == 1:
                transverse_here += 1

        if transverse_here:
            transverse_images += 1
            transverse_instances += transverse_here

    print(f"Malformed annotation files : {len(malformed_labels)}")
    print(f"Invalid class IDs          : {len(invalid_class_ids)}")
    print(f"Invalid bounding boxes     : {len(invalid_coordinates)}")
    print(f"Empty annotation files     : {len(empty_labels)}")
    print()

    print("Class instance counts:")

    for class_id in sorted(VALID_CLASS_IDS):
        print(
            f"  Class {class_id}: "
            f"{class_counts[class_id]}"
        )

    print()

    print(
        f"Images containing transverse cracks : "
        f"{transverse_images}"
    )

    print(
        f"Transverse-crack instances          : "
        f"{transverse_instances}"
    )

    print()

    split_stats[split_name] = {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "unreadable_images": len(unreadable_images),
        "malformed_labels": len(malformed_labels),
        "invalid_class_ids": len(invalid_class_ids),
        "invalid_coordinates": len(invalid_coordinates),
        "empty_labels": len(empty_labels),
        "class_counts": class_counts,
        "transverse_images": transverse_images,
        "transverse_instances": transverse_instances,
    }

    overall_problems.extend(problems)


# ============================================================
# Audit train and validation
# ============================================================

for split, paths in SPLITS.items():

    audit_split(
        split,
        paths["images"],
        paths["labels"]
    )

    print()


# ============================================================
# Cross-split duplicate detection
# ============================================================

print("=" * 72)
print("CROSS-SPLIT DUPLICATE IMAGE CHECK")
print("=" * 72)

cross_split_duplicates = []

for digest, entries in image_hashes.items():

    splits_present = {
        split
        for split, _ in entries
    }

    if len(splits_present) > 1:
        cross_split_duplicates.append(entries)

if cross_split_duplicates:

    print(
        "Exact image duplicates across train/val: "
        f"{len(cross_split_duplicates)}"
    )

    for entries in cross_split_duplicates[:20]:

        print("  Duplicate group:")

        for split, p in entries:
            print(
                f"    {split}: {p.name}"
            )

else:

    print(
        "Exact image duplicates across train/val: 0"
    )

print()


# ============================================================
# Duplicate image content within each split
# ============================================================

print("=" * 72)
print("EXACT DUPLICATE IMAGE CONTENT CHECK")
print("=" * 72)

within_split_duplicate_groups = []

for digest, entries in image_hashes.items():

    for split in {"train", "val"}:

        same_split = [
            (s, p)
            for s, p in entries
            if s == split
        ]

        if len(same_split) > 1:
            within_split_duplicate_groups.append(
                same_split
            )

print(
    "Exact duplicate groups within train/val: "
    f"{len(within_split_duplicate_groups)}"
)

for group in within_split_duplicate_groups[:20]:

    print("  Duplicate group:")

    for split, p in group:
        print(
            f"    {split}: {p.name}"
        )

print()


# ============================================================
# Final summary
# ============================================================

print("=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

train = split_stats.get("train", {})
val = split_stats.get("val", {})

print(
    f"Train images : {train.get('images', 0)}"
)

print(
    f"Train labels : {train.get('labels', 0)}"
)

print(
    f"Val images   : {val.get('images', 0)}"
)

print(
    f"Val labels   : {val.get('labels', 0)}"
)

print()

if (
    train.get("images") == 8464
    and train.get("labels") == 8464
):
    print(
        "✓ Training set count is exactly "
        "8464 image/label pairs."
    )
else:
    print(
        "⚠ Training count differs from expected 8464."
    )

if train.get("images") == train.get("labels"):
    print("✓ Training image/label counts match.")
else:
    print(
        "✗ Training image/label counts DO NOT match."
    )

if val.get("images") == val.get("labels"):
    print("✓ Validation image/label counts match.")
else:
    print(
        "✗ Validation image/label counts DO NOT match."
    )

problem_count = (
    len(overall_problems)
    + len(cross_split_duplicates)
    + len(within_split_duplicate_groups)
)

print()
print("=" * 72)

if problem_count == 0:

    print("RESULT: DATASET AUDIT PASSED")
    print()
    print(
        "No missing labels, orphan labels, corrupt images,"
    )
    print(
        "malformed annotations, invalid boxes, or duplicate"
    )
    print(
        "image content were detected."
    )

else:

    print(
        f"RESULT: DATASET AUDIT FOUND "
        f"{problem_count} ISSUE GROUP(S)"
    )

    print(
        "Review the problems above before training."
    )

print("=" * 72)