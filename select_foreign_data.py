from pathlib import Path
import random
import shutil
import csv

# ============================================================
# RoadVision - Targeted Foreign Dataset Selection
# ============================================================

BASE = Path(r"C:\Users\VICTUS\OneDrive\Documents\RoadVision")
SOURCE = BASE / "foreign_data"
OUTPUT = BASE / "selected_foreign_data"

COUNTRY_TARGETS = {
    "Czech": 250,
    "Japan": 500,
    "Norway": 350,
    "United_States": 400,
}

SEED = 42
TRANSVERSE_FRACTION = 0.60

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
# Read one YOLO label file
# ------------------------------------------------------------

def read_label(label_path):
    classes = []
    instance_counts = {}

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            try:
                cls = int(parts[0])
            except ValueError:
                continue

            # Only RoadVision classes 0-3
            if cls not in CLASS_NAMES:
                continue

            classes.append(cls)
            instance_counts[cls] = instance_counts.get(cls, 0) + 1

    return set(classes), instance_counts


# ------------------------------------------------------------
# Find image corresponding to a label
# ------------------------------------------------------------

def build_image_index(image_dir):
    """Build a filename-stem -> image path lookup once."""
    image_index = {}

    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            image_index[p.stem] = p

    return image_index


# ------------------------------------------------------------
# Load country dataset metadata
# ------------------------------------------------------------

def load_country(country):
    image_dir = SOURCE / country / "images"
    label_dir = SOURCE / country / "labels"

    image_index = build_image_index(image_dir)

    records = []

    for label_path in label_dir.glob("*.txt"):

        image_path = image_index.get(label_path.stem)

        if image_path is None:
            continue

        classes, counts = read_label(label_path)

        if not classes:
            continue

        records.append({
            "country": country,
            "image_path": image_path,
            "label_path": label_path,
            "classes": classes,
            "counts": counts,
        })

    return records

# ------------------------------------------------------------
# Categorize images
# ------------------------------------------------------------

def is_transverse(record):
    return 1 in record["classes"]


def is_multiclass(record):
    return len(record["classes"]) >= 2


def class_signature(record):
    return "+".join(
        str(c) for c in sorted(record["classes"])
    )


# ------------------------------------------------------------
# Select records from a pool without duplicates
# ------------------------------------------------------------

def take_random(pool, n, rng, selected):
    available = [
        r for r in pool
        if id(r) not in selected
    ]

    if n <= 0 or not available:
        return []

    n = min(n, len(available))

    chosen = rng.sample(available, n)

    for r in chosen:
        selected.add(id(r))

    return chosen


# ------------------------------------------------------------
# Main selection logic
# ------------------------------------------------------------

def select_country(records, target, rng):

    selected = set()
    result = []

    transverse_target = round(target * TRANSVERSE_FRACTION)
    diversity_target = target - transverse_target

    # ========================================================
    # PART 1: TRANSVERSE-CRACK SUPPLEMENTATION
    # ========================================================

    transverse_multi = [
        r for r in records
        if is_transverse(r) and is_multiclass(r)
    ]

    transverse_only = [
        r for r in records
        if is_transverse(r) and not is_multiclass(r)
    ]

    # Prefer multi-class transverse images.
    # Roughly 70% of transverse quota comes from these.
    multi_target = round(transverse_target * 0.70)

    chosen = take_random(
        transverse_multi,
        multi_target,
        rng,
        selected
    )

    for r in chosen:
        r["selection_group"] = "transverse_multiclass"
        r["reason"] = "Transverse crack + another damage class"
        result.append(r)

    # Fill remaining transverse quota with transverse images.
    remaining = transverse_target - len(chosen)

    transverse_remaining_pool = [
        r for r in records
        if is_transverse(r) and id(r) not in selected
    ]

    chosen = take_random(
        transverse_remaining_pool,
        remaining,
        rng,
        selected
    )

    for r in chosen:
        r["selection_group"] = "transverse"
        r["reason"] = "Contains Transverse crack (class 1)"
        result.append(r)

    # ========================================================
    # PART 2: NON-TRANSVERSE DIVERSITY
    # ========================================================

    non_transverse = [
        r for r in records
        if not is_transverse(r)
    ]

    # Prefer multi-class images first.
    nontrans_multi = [
        r for r in non_transverse
        if is_multiclass(r)
    ]

    multi_diversity_target = round(diversity_target * 0.50)

    chosen = take_random(
        nontrans_multi,
        multi_diversity_target,
        rng,
        selected
    )

    for r in chosen:
        r["selection_group"] = "diversity_multiclass"
        r["reason"] = "Multi-class damage diversity"
        result.append(r)

    # Remaining slots: balance among individual classes.
    remaining = diversity_target - len(chosen)

    single_class_pools = {
        0: [
            r for r in non_transverse
            if r["classes"] == {0}
        ],
        2: [
            r for r in non_transverse
            if r["classes"] == {2}
        ],
        3: [
            r for r in non_transverse
            if r["classes"] == {3}
        ],
    }

    # Divide remaining slots between the three non-transverse
    # classes as evenly as possible.
    base_each = remaining // 3
    extra = remaining % 3

    class_order = [0, 2, 3]

    for i, cls in enumerate(class_order):

        quota = base_each

        if i < extra:
            quota += 1

        chosen = take_random(
            single_class_pools[cls],
            quota,
            rng,
            selected
        )

        for r in chosen:
            r["selection_group"] = "single_class_diversity"
            r["reason"] = f"{CLASS_NAMES[cls]} class diversity"
            result.append(r)

    # ========================================================
    # PART 3: FILL ANY REMAINING SLOTS
    # ========================================================

    if len(result) < target:

        remaining_pool = [
            r for r in records
            if id(r) not in selected
        ]

        chosen = take_random(
            remaining_pool,
            target - len(result),
            rng,
            selected
        )

        for r in chosen:
            r["selection_group"] = "fill"
            r["reason"] = "Fill remaining country quota"
            result.append(r)

    return result


# ------------------------------------------------------------
# Prepare output directories
# ------------------------------------------------------------

output_images = OUTPUT / "images"
output_labels = OUTPUT / "labels"

output_images.mkdir(parents=True, exist_ok=True)
output_labels.mkdir(parents=True, exist_ok=True)

# Do not silently delete old files.
# This makes accidental data loss less likely.


# ------------------------------------------------------------
# Selection
# ------------------------------------------------------------

rng = random.Random(SEED)

all_selected = []

print("=" * 70)
print("RoadVision Targeted Foreign Dataset Selection")
print("=" * 70)
print(f"Random seed: {SEED}")
print()

for country, target in COUNTRY_TARGETS.items():

    records = load_country(country)

    print(f"{country}:")
    print(f"  Available: {len(records)}")
    print(f"  Target:    {target}")

    if len(records) < target:
        raise RuntimeError(
            f"{country} has only {len(records)} usable images "
            f"but {target} were requested."
        )

    selected = select_country(
        records,
        target,
        rng
    )

    print(f"  Selected:  {len(selected)}")

    all_selected.extend(selected)

    print()


# ------------------------------------------------------------
# Copy selected files
# ------------------------------------------------------------

manifest_path = OUTPUT / "selection_manifest.csv"

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "country",
        "original_image",
        "selected_image",
        "classes",
        "class_names",
        "instance_counts",
        "selection_group",
        "reason",
    ])

    for record in all_selected:

        country = record["country"]
        image_path = record["image_path"]
        label_path = record["label_path"]

        # Country prefix prevents filename collisions.
        new_stem = f"{country}_{image_path.stem}"

        new_image_name = (
            new_stem + image_path.suffix.lower()
        )

        new_label_name = (
            new_stem + ".txt"
        )

        destination_image = (
            output_images / new_image_name
        )

        destination_label = (
            output_labels / new_label_name
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        shutil.copy2(
            label_path,
            destination_label
        )

        class_ids = sorted(record["classes"])

        class_names = [
            CLASS_NAMES[c]
            for c in class_ids
        ]

        instance_counts = {
            CLASS_NAMES[c]: record["counts"].get(c, 0)
            for c in class_ids
        }

        writer.writerow([
            country,
            image_path.name,
            new_image_name,
            "+".join(map(str, class_ids)),
            "+".join(class_names),
            str(instance_counts),
            record["selection_group"],
            record["reason"],
        ])


# ------------------------------------------------------------
# Final statistics
# ------------------------------------------------------------

print("=" * 70)
print("SELECTION COMPLETE")
print("=" * 70)

print(f"Total selected images: {len(all_selected)}")
print(f"Output: {OUTPUT}")
print(f"Manifest: {manifest_path}")
print()

total_instances = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
}

total_images = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
}

for record in all_selected:

    for cls, count in record["counts"].items():
        total_instances[cls] += count

    for cls in record["classes"]:
        total_images[cls] += 1

print("Selected foreign-data distribution:")
print()

for cls in range(4):
    print(
        f"  Class {cls} ({CLASS_NAMES[cls]}): "
        f"{total_instances[cls]} instances / "
        f"{total_images[cls]} images"
    )

print()
print("By country:")

for country in COUNTRY_TARGETS:
    count = sum(
        1 for r in all_selected
        if r["country"] == country
    )

    print(f"  {country}: {count}")

print()
print("You can now inspect selected_foreign_data before")
print("merging it into the RoadVision training set.")