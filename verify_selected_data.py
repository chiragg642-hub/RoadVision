from pathlib import Path
from collections import Counter

# ============================================================
# RoadVision - Verify Selected Foreign Dataset
# ============================================================

BASE = Path(r"C:\Users\VICTUS\OneDrive\Documents\RoadVision")

INDIA_IMAGES = BASE / "dataset" / "images" / "train"
INDIA_LABELS = BASE / "dataset" / "labels" / "train"

FOREIGN_IMAGES = BASE / "selected_foreign_data" / "images"
FOREIGN_LABELS = BASE / "selected_foreign_data" / "labels"

CLASS_NAMES = {
    0: "Longitudinal",
    1: "Transverse",
    2: "Alligator",
    3: "Pothole",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ------------------------------------------------------------
# Get images
# ------------------------------------------------------------

def get_images(folder):
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ------------------------------------------------------------
# Check labels
# ------------------------------------------------------------

def analyze_dataset(image_dir, label_dir, dataset_name):

    print()
    print("=" * 70)
    print(f"{dataset_name}")
    print("=" * 70)

    images = get_images(image_dir)
    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")

    print()
    print("Matching:")
    print(f"  Images without labels: {len(missing_labels)}")
    print(f"  Labels without images: {len(orphan_labels)}")

    if missing_labels:
        print("\nFirst missing labels:")
        for x in sorted(missing_labels)[:10]:
            print(f"  {x}")

    if orphan_labels:
        print("\nFirst orphan labels:")
        for x in sorted(orphan_labels)[:10]:
            print(f"  {x}")

    # --------------------------------------------------------
    # Validate YOLO labels
    # --------------------------------------------------------

    class_counts = Counter()
    image_class_counts = Counter()

    invalid_lines = []
    invalid_coordinates = []

    for label_path in labels:

        classes_in_image = set()

        with open(label_path, "r", encoding="utf-8") as f:

            for line_number, line in enumerate(f, start=1):

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                # YOLO format:
                # class x_center y_center width height
                if len(parts) != 5:
                    invalid_lines.append(
                        (label_path.name, line_number, line)
                    )
                    continue

                try:
                    cls = int(parts[0])

                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:
                    invalid_lines.append(
                        (label_path.name, line_number, line)
                    )
                    continue

                # Class must be 0-3
                if cls not in CLASS_NAMES:
                    invalid_lines.append(
                        (label_path.name, line_number, line)
                    )
                    continue

                # YOLO normalized coordinates must be 0-1
                if not (
                    0 <= x <= 1 and
                    0 <= y <= 1 and
                    0 < w <= 1 and
                    0 < h <= 1
                ):
                    invalid_coordinates.append(
                        (label_path.name, line_number, line)
                    )
                    continue

                class_counts[cls] += 1
                classes_in_image.add(cls)

        for cls in classes_in_image:
            image_class_counts[cls] += 1

    print()
    print("YOLO label validation:")
    print(f"  Invalid label lines: {len(invalid_lines)}")
    print(f"  Invalid coordinates: {len(invalid_coordinates)}")

    if invalid_lines:
        print("\nFirst invalid lines:")
        for item in invalid_lines[:10]:
            print(f"  {item}")

    if invalid_coordinates:
        print("\nFirst invalid coordinates:")
        for item in invalid_coordinates[:10]:
            print(f"  {item}")

    print()
    print("Class distribution:")

    for cls in range(4):
        print(
            f"  Class {cls} ({CLASS_NAMES[cls]}): "
            f"{class_counts[cls]} instances / "
            f"{image_class_counts[cls]} images"
        )

    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "invalid_lines": len(invalid_lines),
        "invalid_coordinates": len(invalid_coordinates),
        "instances": class_counts,
        "image_counts": image_class_counts,
    }


# ============================================================
# Analyze India
# ============================================================

india = analyze_dataset(
    INDIA_IMAGES,
    INDIA_LABELS,
    "INDIA TRAINING DATA"
)


# ============================================================
# Analyze selected foreign data
# ============================================================

foreign = analyze_dataset(
    FOREIGN_IMAGES,
    FOREIGN_LABELS,
    "SELECTED FOREIGN DATA"
)


# ============================================================
# Combined distribution
# ============================================================

print()
print("=" * 70)
print("COMBINED INDIA + SELECTED FOREIGN TRAINING DATA")
print("=" * 70)

combined_instances = Counter()
combined_images = Counter()

for cls in range(4):
    combined_instances[cls] = (
        india["instances"][cls]
        + foreign["instances"][cls]
    )

    combined_images[cls] = (
        india["image_counts"][cls]
        + foreign["image_counts"][cls]
    )

print()

for cls in range(4):
    print(
        f"Class {cls} ({CLASS_NAMES[cls]}): "
        f"{combined_instances[cls]} instances / "
        f"{combined_images[cls]} images"
    )


# ============================================================
# Class ratios
# ============================================================

print()
print("Class balance:")

values = [
    combined_instances[cls]
    for cls in range(4)
]

smallest = min(values)
largest = max(values)

print(f"  Smallest class: {smallest}")
print(f"  Largest class:  {largest}")
print(f"  Largest / smallest ratio: {largest / smallest:.2f}x")


# ============================================================
# Overall verdict
# ============================================================

print()
print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

problems = []

if foreign["images"] != 1500:
    problems.append(
        f"Expected 1500 foreign images, found {foreign['images']}"
    )

if foreign["missing_labels"] != 0:
    problems.append(
        f"{foreign['missing_labels']} foreign images have no labels"
    )

if foreign["orphan_labels"] != 0:
    problems.append(
        f"{foreign['orphan_labels']} foreign labels have no images"
    )

if foreign["invalid_lines"] != 0:
    problems.append(
        f"{foreign['invalid_lines']} invalid YOLO label lines"
    )

if foreign["invalid_coordinates"] != 0:
    problems.append(
        f"{foreign['invalid_coordinates']} invalid YOLO coordinates"
    )


if problems:

    print("⚠️ PROBLEMS FOUND:")
    for problem in problems:
        print(f"  - {problem}")

else:

    print("✓ Foreign dataset contains exactly 1500 images")
    print("✓ Every foreign image has a matching label")
    print("✓ Every foreign label has a matching image")
    print("✓ YOLO label format is valid")
    print("✓ All classes are within RoadVision classes 0-3")
    print("✓ All bounding-box coordinates are valid")
    print("✓ Combined class distribution calculated successfully")

print()
print("NO FILES WERE MODIFIED.")
print("NO DATA WAS MERGED.")