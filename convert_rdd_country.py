import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import sys

# --------------------------------------------------
# Usage:
# python convert_rdd_country.py "COUNTRY" "SOURCE_DIR" "OUTPUT_DIR"
#
# Example:
# python convert_rdd_country.py Czech "D:\...\Czech\train" "C:\...\RoadVision\foreign_data\Czech"
# --------------------------------------------------

if len(sys.argv) != 4:
    print("Usage: python convert_rdd_country.py COUNTRY SOURCE_DIR OUTPUT_DIR")
    sys.exit(1)

country = sys.argv[1]
source_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

xml_dir = source_dir / "annotations" / "xmls"
image_dir = source_dir / "images"

out_images = output_dir / "images"
out_labels = output_dir / "labels"

out_images.mkdir(parents=True, exist_ok=True)
out_labels.mkdir(parents=True, exist_ok=True)

# RDD2022 → RoadVision mapping
CLASS_MAP = {
    "D00": 0,
    "D01": 0,
    "D0w0": 0,

    "D10": 1,
    "D11": 1,

    "D20": 2,

    "D40": 3,
}

stats = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
}

images_with_class = {
    0: set(),
    1: set(),
    2: set(),
    3: set(),
}

converted_images = 0
skipped_images = 0

for xml_file in xml_dir.glob("*.xml"):

    try:
        root = ET.parse(xml_file).getroot()
    except Exception as e:
        print(f"Could not parse {xml_file}: {e}")
        continue

    image_name = root.findtext("filename")

    if not image_name:
        image_name = xml_file.stem + ".jpg"

    source_image = image_dir / image_name

    if not source_image.exists():
        # Try common extensions
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = image_dir / (xml_file.stem + ext)
            if candidate.exists():
                source_image = candidate
                image_name = candidate.name
                break

    objects = []

    for obj in root.findall("object"):
        original_class = obj.findtext("name")

        if original_class is None:
            continue

        original_class = original_class.strip()

        if original_class not in CLASS_MAP:
            continue

        roadvision_class = CLASS_MAP[original_class]

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        size = root.find("size")

        if size is None:
            continue

        width = float(size.findtext("width"))
        height = float(size.findtext("height"))

        # Convert Pascal VOC → YOLO
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        # Clamp values just in case
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        box_width = max(0, min(1, box_width))
        box_height = max(0, min(1, box_height))

        objects.append(
            f"{roadvision_class} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        stats[roadvision_class] += 1
        images_with_class[roadvision_class].add(xml_file.stem)

    # If an image contains none of our four target classes,
    # don't copy it into the RoadVision dataset.
    if not objects:
        skipped_images += 1
        continue

    if not source_image.exists():
        print(f"WARNING: Image not found for {xml_file.name}")
        continue

    destination_image = out_images / source_image.name
    destination_label = out_labels / f"{source_image.stem}.txt"

    shutil.copy2(source_image, destination_image)

    with open(destination_label, "w") as f:
        f.write("\n".join(objects) + "\n")

    converted_images += 1


print()
print("=" * 50)
print(f"Country: {country}")
print("=" * 50)

print(f"Converted images: {converted_images}")
print(f"Skipped images:   {skipped_images}")

print()
print("RoadVision class distribution:")

names = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole",
}

for cls in range(4):
    print(
        f"class {cls} ({names[cls]}): "
        f"{stats[cls]} instances / "
        f"{len(images_with_class[cls])} images"
    )