from pathlib import Path
from collections import Counter
from itertools import combinations

BASE = Path(r"C:\Users\VICTUS\OneDrive\Documents\RoadVision\foreign_data")

CLASS_NAMES = {
    0: "Longitudinal",
    1: "Transverse",
    2: "Alligator",
    3: "Pothole",
}

COUNTRIES = [
    "Czech",
    "Japan",
    "Norway",
    "United_States",
]


def analyze_country(country):
    label_dir = BASE / country / "labels"

    image_count = 0
    class_images = Counter()
    class_instances = Counter()
    combinations_count = Counter()

    for label_file in label_dir.glob("*.txt"):

        classes_in_image = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                cls = int(parts[0])

                class_instances[cls] += 1

                if cls not in classes_in_image:
                    classes_in_image.append(cls)

        if not classes_in_image:
            continue

        image_count += 1

        # Count each class once per image
        for cls in classes_in_image:
            class_images[cls] += 1

        # Count class combinations
        classes_in_image.sort()

        if len(classes_in_image) >= 2:
            for combo in combinations(classes_in_image, 2):
                combinations_count[combo] += 1

    print()
    print("=" * 65)
    print(f"{country}")
    print("=" * 65)

    print(f"Images containing RoadVision damage: {image_count}")

    print()
    print("IMAGE COUNTS BY CLASS")
    print("-" * 65)

    for cls in range(4):
        print(
            f"{cls} ({CLASS_NAMES[cls]:12}): "
            f"{class_images[cls]:5} images | "
            f"{class_instances[cls]:5} instances"
        )

    print()
    print("CLASS COMBINATIONS")
    print("-" * 65)

    for (a, b), count in sorted(
        combinations_count.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            f"{CLASS_NAMES[a]:12} + "
            f"{CLASS_NAMES[b]:12}: "
            f"{count:5} images"
        )


for country in COUNTRIES:
    analyze_country(country)