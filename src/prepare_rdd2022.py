from pathlib import Path
import xml.etree.ElementTree as ET
import random
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

RDD_ROOT = Path(r"C:\Users\VICTUS\Downloads\RDD2022_released_through_CRDDC2022\India")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "dataset"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Official four-class CRDDC2022 taxonomy
CLASS_MAP = {
    "D00": 0,  # Longitudinal crack
    "D10": 1,  # Transverse crack
    "D20": 2,  # Alligator crack
    "D40": 3,  # Pothole
}


# ============================================================
# SOURCE
# ============================================================

SOURCE_IMAGES = RDD_ROOT / "train" / "images"
SOURCE_XMLS = RDD_ROOT / "train" / "annotations" / "xmls"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

TRAIN_IMAGES = OUTPUT_ROOT / "images" / "train"
VAL_IMAGES = OUTPUT_ROOT / "images" / "val"

TRAIN_LABELS = OUTPUT_ROOT / "labels" / "train"
VAL_LABELS = OUTPUT_ROOT / "labels" / "val"


for directory in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# XML -> YOLO
# ============================================================

def convert_xml_to_yolo(xml_file, output_txt):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")

    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):

        class_name = obj.find("name").text.strip()

        # Ignore classes outside the four RoadVision classes
        if class_name not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[class_name]

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Clamp coordinates to image boundaries
        xmin = max(0, min(xmin, image_width))
        xmax = max(0, min(xmax, image_width))
        ymin = max(0, min(ymin, image_height))
        ymax = max(0, min(ymax, image_height))

        # Skip invalid boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        # Pascal VOC -> YOLO
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height

        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    output_txt.write_text(
        "\n".join(yolo_lines),
        encoding="utf-8"
    )


# ============================================================
# FIND IMAGES
# ============================================================

image_files = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:
    image_files.extend(SOURCE_IMAGES.glob(extension))

image_files = sorted(image_files)

print(f"Found {len(image_files)} images.")

if not image_files:
    raise RuntimeError(
        f"No images found in:\n{SOURCE_IMAGES}"
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

random.seed(RANDOM_SEED)
random.shuffle(image_files)

split_index = int(len(image_files) * TRAIN_RATIO)

train_files = image_files[:split_index]
val_files = image_files[split_index:]

print(f"Training images:   {len(train_files)}")
print(f"Validation images: {len(val_files)}")


# ============================================================
# PROCESS
# ============================================================

def process_images(files, image_output_dir, label_output_dir):

    processed = 0
    skipped = 0

    for image_file in files:

        xml_file = SOURCE_XMLS / f"{image_file.stem}.xml"

        if not xml_file.exists():
            print(f"WARNING: Missing annotation: {xml_file.name}")
            skipped += 1
            continue

        destination_image = image_output_dir / image_file.name

        shutil.copy2(
            image_file,
            destination_image
        )

        destination_label = (
            label_output_dir /
            f"{image_file.stem}.txt"
        )

        convert_xml_to_yolo(
            xml_file,
            destination_label
        )

        processed += 1

    return processed, skipped


print("\nProcessing training data...")

train_processed, train_skipped = process_images(
    train_files,
    TRAIN_IMAGES,
    TRAIN_LABELS,
)


print("\nProcessing validation data...")

val_processed, val_skipped = process_images(
    val_files,
    VAL_IMAGES,
    VAL_LABELS,
)


# ============================================================
# DATA.YAML
# ============================================================

yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val

names:
  0: longitudinal_crack
  1: transverse_crack
  2: alligator_crack
  3: pothole
"""

(OUTPUT_ROOT / "data.yaml").write_text(
    yaml_content,
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n======================================")
print("RDD2022 preparation complete!")
print("======================================")

print(f"Train processed: {train_processed}")
print(f"Train skipped:   {train_skipped}")

print(f"Val processed:   {val_processed}")
print(f"Val skipped:     {val_skipped}")

print("\nClasses:")
print("0 = longitudinal_crack")
print("1 = transverse_crack")
print("2 = alligator_crack")
print("3 = pothole")

print(f"\nDataset created at:")
print(OUTPUT_ROOT)

print("\ndata.yaml created successfully.")