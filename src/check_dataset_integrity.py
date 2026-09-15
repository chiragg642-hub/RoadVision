from pathlib import Path
from PIL import Image

# RoadVision dataset
DATASET = Path(__file__).resolve().parent.parent / "dataset"

SPLITS = ["train", "val"]
NUM_CLASSES = 4

errors = []
warnings = []

print("=" * 60)
print("RoadVision Dataset Integrity Check")
print("=" * 60)

for split in SPLITS:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    print(f"\nChecking {split.upper()}...")

    if not image_dir.exists():
        errors.append(f"Missing image directory: {image_dir}")
        continue

    if not label_dir.exists():
        errors.append(f"Missing label directory: {label_dir}")
        continue

    images = {
        p.stem: p
        for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    labels = {
        p.stem: p
        for p in label_dir.iterdir()
        if p.suffix.lower() == ".txt"
    }

    print(f"Images found: {len(images)}")
    print(f"Labels found: {len(labels)}")

    # --------------------------------------------------
    # 1. Every image should have a label file
    # --------------------------------------------------
    missing_labels = set(images) - set(labels)

    if missing_labels:
        for name in sorted(missing_labels):
            errors.append(
                f"{split}: image has no label file: {name}"
            )

    # --------------------------------------------------
    # 2. Every label should have an image
    # --------------------------------------------------
    orphan_labels = set(labels) - set(images)

    if orphan_labels:
        for name in sorted(orphan_labels):
            errors.append(
                f"{split}: label has no corresponding image: {name}"
            )

    # --------------------------------------------------
    # 3. Validate image files
    # --------------------------------------------------
    bad_images = 0

    for name, image_path in images.items():
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            bad_images += 1
            errors.append(
                f"{split}: invalid image {image_path.name}: {e}"
            )

    print(f"Invalid images: {bad_images}")

    # --------------------------------------------------
    # 4. Validate YOLO labels
    # --------------------------------------------------
    invalid_labels = 0
    total_boxes = 0
    empty_labels = 0

    for name, label_path in labels.items():

        try:
            content = label_path.read_text(encoding="utf-8").strip()

            # Empty label = background image
            if not content:
                empty_labels += 1
                continue

            for line_number, line in enumerate(
                content.splitlines(), start=1
            ):

                parts = line.split()

                # YOLO format:
                # class x_center y_center width height
                if len(parts) != 5:
                    invalid_labels += 1
                    errors.append(
                        f"{split}: {label_path.name}, "
                        f"line {line_number}: expected 5 values"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                except ValueError:
                    invalid_labels += 1
                    errors.append(
                        f"{split}: {label_path.name}, "
                        f"line {line_number}: non-numeric value"
                    )
                    continue

                # Class ID
                if not (0 <= class_id < NUM_CLASSES):
                    invalid_labels += 1
                    errors.append(
                        f"{split}: {label_path.name}, "
                        f"line {line_number}: "
                        f"invalid class ID {class_id}"
                    )

                # YOLO coordinates must be 0..1
                values = {
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }

                for field, value in values.items():
                    if not (0 <= value <= 1):
                        invalid_labels += 1
                        errors.append(
                            f"{split}: {label_path.name}, "
                            f"line {line_number}: "
                            f"{field}={value} outside [0,1]"
                        )

                # Bounding box must have positive dimensions
                if width <= 0 or height <= 0:
                    invalid_labels += 1
                    errors.append(
                        f"{split}: {label_path.name}, "
                        f"line {line_number}: "
                        f"non-positive box size"
                    )

                total_boxes += 1

        except Exception as e:
            invalid_labels += 1
            errors.append(
                f"{split}: could not read {label_path.name}: {e}"
            )

    print(f"Total bounding boxes: {total_boxes}")
    print(f"Empty label files: {empty_labels}")
    print(f"Invalid label entries: {invalid_labels}")


# ======================================================
# 5. Check train/validation overlap
# ======================================================

train_dir = DATASET / "images" / "train"
val_dir = DATASET / "images" / "val"

train_names = {
    p.stem
    for p in train_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
}

val_names = {
    p.stem
    for p in val_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
}

overlap = train_names & val_names

print("\nChecking train/validation overlap...")
print(f"Overlapping image names: {len(overlap)}")

if overlap:
    for name in sorted(overlap):
        errors.append(
            f"Image appears in both train and validation: {name}"
        )


# ======================================================
# Final result
# ======================================================

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

if errors:
    print(f"\n❌ DATASET CHECK FAILED")
    print(f"Errors found: {len(errors)}\n")

    for error in errors[:50]:
        print(" -", error)

    if len(errors) > 50:
        print(f"\n... and {len(errors) - 50} more errors.")

else:
    print("\n✅ DATASET INTEGRITY CHECK PASSED")
    print("\nThe dataset is structurally valid.")
    print("No conversion errors were detected.")
    print("No train/validation overlap was detected.")
    print("\nRoadVision is ready for training.")