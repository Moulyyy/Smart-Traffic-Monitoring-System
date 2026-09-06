import os
import cv2
import shutil
from pathlib import Path

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent
data_dir = ROOT_DIR / "data"
output_dir = ROOT_DIR / "data" / "yolo_dataset"

splits = ['training', 'testing']

# Create output structure
for split in ['train', 'val']:
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

# Class mapping
class_map = {'Gallardo' : 2}  # Add more classes if found

def convert_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height

# Process both splits
for split in splits:
    label_file = os.path.join(data_dir, split, "bounding_boxes.labels")
    image_dir = os.path.join(data_dir, split)
    output_split = 'train' if split == 'training' else 'val'

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().rsplit(' ', 1)
            if len(parts) != 2:
                continue
            bbox_info, class_name = parts
            elems = bbox_info.split()
            if len(elems) != 5:
                continue
            filename, x_min, y_min, x_max, y_max = elems
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

            img_path = os.path.join(image_dir, filename)
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # Convert bbox
            x_center, y_center, width, height = convert_to_yolo((x_min, y_min, x_max, y_max), w, h)

            # Write label file
            class_id = class_map.get(class_name)
            if class_id is None:
                continue

            label_txt = f"{output_dir}/labels/{output_split}/{Path(filename).stem}.txt"
            with open(label_txt, 'a') as lf:
                lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Copy image
            shutil.copy(img_path, f"{output_dir}/images/{output_split}/{filename}")

print("✅ Conversion complete. YOLO-format data saved to:", output_dir)
