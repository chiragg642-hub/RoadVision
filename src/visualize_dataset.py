from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "dataset"

IMAGE_DIR = DATASET / "images" / "train"
LABEL_DIR = DATASET / "labels" / "train"

OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_preview"

NUM_IMAGES = 10
RANDOM_SEED = 42


CLASS_NAMES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "pothole",
}


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)


# ============================================================
# FIND IMAGES
# ============================================================

image_files = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:
    image_files.extend(IMAGE_DIR.glob(extension))

if not image_files:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")


# ============================================================
# READ YOLO LABELS
# ============================================================

def read_labels(label_file, image_width, image_height):

    objects = []

    if not label_file.exists():
        return objects

    text = label_file.read_text(encoding="utf-8").strip()

    if not text:
        return objects

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # YOLO normalized coordinates -> pixels

        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2

        objects.append(
            (
                class_id,
                xmin,
                ymin,
                xmax,
                ymax,
            )
        )

    return objects


# ============================================================
# SELECT DIFFERENT TYPES OF EXAMPLES
# ============================================================

labelled_images = []
empty_images = []

for image_file in image_files:

    label_file = LABEL_DIR / f"{image_file.stem}.txt"

    if not label_file.exists():
        continue

    if label_file.read_text(encoding="utf-8").strip():
        labelled_images.append(image_file)
    else:
        empty_images.append(image_file)


# Pick mostly labelled examples
selected = random.sample(
    labelled_images,
    min(NUM_IMAGES - 2, len(labelled_images))
)

# Add two empty/background images
if empty_images:
    selected.extend(
        random.sample(
            empty_images,
            min(2, len(empty_images))
        )
    )


# ============================================================
# DRAW BOXES
# ============================================================

for index, image_file in enumerate(selected, start=1):

    image = Image.open(image_file).convert("RGB")

    image_width, image_height = image.size

    draw = ImageDraw.Draw(image)

    label_file = LABEL_DIR / f"{image_file.stem}.txt"

    objects = read_labels(
        label_file,
        image_width,
        image_height,
    )

    for class_id, xmin, ymin, xmax, ymax in objects:

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline="red",
            width=4,
        )

        draw.text(
            (xmin, max(0, ymin - 20)),
            class_name,
            fill="red",
        )

    output_file = (
        OUTPUT_DIR /
        f"{index:02d}_{image_file.name}"
    )

    image.save(output_file)

    print(f"Saved: {output_file}")


print("\n======================================")
print("Visualization complete!")
print("======================================")

print(f"Images saved to:")
print(OUTPUT_DIR)