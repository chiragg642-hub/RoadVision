from pathlib import Path
from collections import Counter

DATASET = Path(__file__).resolve().parent.parent / "dataset"

TRAIN_LABELS = DATASET / "labels" / "train"
VAL_LABELS = DATASET / "labels" / "val"

CLASS_NAMES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "pothole",
}

def analyze(label_dir):
    class_counts = Counter()
    total_boxes = 0
    empty_files = 0

    for label_file in label_dir.glob("*.txt"):

        text = label_file.read_text(encoding="utf-8").strip()

        if not text:
            empty_files += 1
            continue

        for line in text.splitlines():

            parts = line.split()

            if len(parts) != 5:
                print(f"WARNING: Invalid label: {label_file.name}")
                continue

            class_id = int(parts[0])

            class_counts[class_id] += 1
            total_boxes += 1

    return class_counts, total_boxes, empty_files


for name, directory in [
    ("TRAIN", TRAIN_LABELS),
    ("VALIDATION", VAL_LABELS),
]:

    counts, total, empty = analyze(directory)

    print("\n================================")
    print(name)
    print("================================")

    print(f"Total bounding boxes: {total}")
    print(f"Images with no retained damage: {empty}")

    for class_id in range(4):
        print(
            f"{class_id} - "
            f"{CLASS_NAMES[class_id]}: "
            f"{counts[class_id]}"
        )