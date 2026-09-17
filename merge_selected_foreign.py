from pathlib import Path
import shutil
from collections import Counter

# ============================================================
# RoadVision - Merge Selected Foreign Data into Training Set
# ============================================================

BASE = Path(r"C:\Users\VICTUS\OneDrive\Documents\RoadVision")

SOURCE_IMAGES = BASE / "selected_foreign_data" / "images"
SOURCE_LABELS = BASE / "selected_foreign_data" / "labels"

TRAIN_IMAGES = BASE / "dataset" / "images" / "train"
TRAIN_LABELS = BASE / "dataset" / "labels" / "train"

CLASS_NAMES = {
    0: "Longitudinal",
    1: "Transverse",
    2: "Alligator",
    3: "Pothole",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ============================================================
# Helper functions
# ============================================================

def get_images(folder):
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_labels(folder):
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ]


def get_stem_set(files):
    return {p.stem for p in files}


def count_classes(label_files):
    counts = Counter()

    for label_path in label_files:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                try:
                    cls = int(parts[0])
                except ValueError:
                    continue

                if cls in CLASS_NAMES:
                    counts[cls] += 1

    return counts


def count_images_per_class(label_files):
    counts = Counter()

    for label_path in label_files:

        classes_in_image = set()

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                try:
                    cls = int(parts[0])
                except ValueError:
                    continue

                if cls in CLASS_NAMES:
                    classes_in_image.add(cls)

        for cls in classes_in_image:
            counts[cls] += 1

    return counts


# ============================================================
# Make sure directories exist
# ============================================================

if not SOURCE_IMAGES.exists():
    raise RuntimeError(
        f"Source image directory does not exist:\n{SOURCE_IMAGES}"
    )

if not SOURCE_LABELS.exists():
    raise RuntimeError(
        f"Source label directory does not exist:\n{SOURCE_LABELS}"
    )

if not TRAIN_IMAGES.exists():
    raise RuntimeError(
        f"Training image directory does not exist:\n{TRAIN_IMAGES}"
    )

if not TRAIN_LABELS.exists():
    raise RuntimeError(
        f"Training label directory does not exist:\n{TRAIN_LABELS}"
    )


# ============================================================
# Read source data
# ============================================================

source_images = get_images(SOURCE_IMAGES)
source_labels = get_labels(SOURCE_LABELS)

train_images = get_images(TRAIN_IMAGES)
train_labels = get_labels(TRAIN_LABELS)

print("=" * 70)
print("RoadVision - Merge Selected Foreign Data")
print("=" * 70)

print()
print("SOURCE:")
print(f"  Images: {len(source_images)}")
print(f"  Labels: {len(source_labels)}")

print()
print("CURRENT TRAINING SET:")
print(f"  Images: {len(train_images)}")
print(f"  Labels: {len(train_labels)}")


# ============================================================
# Verify source image/label matching
# ============================================================

source_image_stems = get_stem_set(source_images)
source_label_stems = get_stem_set(source_labels)

missing_source_labels = source_image_stems - source_label_stems
orphan_source_labels = source_label_stems - source_image_stems

if missing_source_labels:
    raise RuntimeError(
        f"Source has {len(missing_source_labels)} images without labels."
    )

if orphan_source_labels:
    raise RuntimeError(
        f"Source has {len(orphan_source_labels)} labels without images."
    )


# ============================================================
# Check filename collisions BEFORE copying anything
# ============================================================

train_image_stems = get_stem_set(train_images)
train_label_stems = get_stem_set(train_labels)

image_collisions = source_image_stems & train_image_stems
label_collisions = source_label_stems & train_label_stems

print()
print("COLLISION CHECK:")
print(f"  Image filename collisions: {len(image_collisions)}")
print(f"  Label filename collisions: {len(label_collisions)}")

if image_collisions:
    print("\nFirst image collisions:")
    for stem in sorted(image_collisions)[:10]:
        print(f"  {stem}")

    raise RuntimeError(
        "Image filename collisions detected. "
        "Nothing was copied."
    )

if label_collisions:
    print("\nFirst label collisions:")
    for stem in sorted(label_collisions)[:10]:
        print(f"  {stem}")

    raise RuntimeError(
        "Label filename collisions detected. "
        "Nothing was copied."
    )


# ============================================================
# Confirm source counts
# ============================================================

if len(source_images) != 1500:
    raise RuntimeError(
        f"Expected 1500 selected foreign images, "
        f"but found {len(source_images)}. Nothing was copied."
    )

if len(source_labels) != 1500:
    raise RuntimeError(
        f"Expected 1500 selected foreign labels, "
        f"but found {len(source_labels)}. Nothing was copied."
    )


# ============================================================
# Display source distribution
# ============================================================

source_instance_counts = count_classes(source_labels)
source_image_counts = count_images_per_class(source_labels)

print()
print("FOREIGN DATA TO BE ADDED:")

for cls in range(4):
    print(
        f"  Class {cls} ({CLASS_NAMES[cls]}): "
        f"{source_instance_counts[cls]} instances / "
        f"{source_image_counts[cls]} images"
    )


# ============================================================
# COPY
# ============================================================

print()
print("=" * 70)
print("COPYING 1500 FOREIGN IMAGES + LABELS")
print("=" * 70)

copied_images = 0
copied_labels = 0

for image_path in source_images:

    destination = TRAIN_IMAGES / image_path.name

    shutil.copy2(
        image_path,
        destination
    )

    copied_images += 1


for label_path in source_labels:

    destination = TRAIN_LABELS / label_path.name

    shutil.copy2(
        label_path,
        destination
    )

    copied_labels += 1


# ============================================================
# Verify final dataset
# ============================================================

final_images = get_images(TRAIN_IMAGES)
final_labels = get_labels(TRAIN_LABELS)

final_image_stems = get_stem_set(final_images)
final_label_stems = get_stem_set(final_labels)

missing_final_labels = final_image_stems - final_label_stems
orphan_final_labels = final_label_stems - final_image_stems

final_instance_counts = count_classes(final_labels)
final_image_counts = count_images_per_class(final_labels)


# ============================================================
# Final report
# ============================================================

print()
print("=" * 70)
print("MERGE COMPLETE")
print("=" * 70)

print()
print("Copied:")
print(f"  Images: {copied_images}")
print(f"  Labels: {copied_labels}")

print()
print("FINAL TRAINING SET:")
print(f"  Images: {len(final_images)}")
print(f"  Labels: {len(final_labels)}")

print()
print("FINAL CLASS DISTRIBUTION:")

for cls in range(4):
    print(
        f"  Class {cls} ({CLASS_NAMES[cls]}): "
        f"{final_instance_counts[cls]} instances / "
        f"{final_image_counts[cls]} images"
    )

print()
print("FINAL MATCHING CHECK:")
print(f"  Images without labels: {len(missing_final_labels)}")
print(f"  Labels without images: {len(orphan_final_labels)}")

if (
    copied_images == 1500
    and copied_labels == 1500
    and len(missing_final_labels) == 0
    and len(orphan_final_labels) == 0
):
    print()
    print("✓ MERGE VERIFIED SUCCESSFULLY")
    print("✓ 1500 foreign images added")
    print("✓ 1500 foreign labels added")
    print("✓ Every training image has a matching label")
    print("✓ Validation dataset was NOT modified")

else:
    print()
    print("⚠️ SOMETHING NEEDS TO BE CHECKED.")

print()
print("IMPORTANT:")
print("  dataset/images/val was NOT touched.")
print("  dataset/labels/val was NOT touched.")
print("  foreign_data was NOT touched.")
print("  selected_foreign_data was NOT touched.")