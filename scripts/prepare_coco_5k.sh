#!/usr/bin/env bash
set -e

# Create directory
mkdir -p data/coco2017
cd data/coco2017

# Step 1: Download val2017 images
echo "Downloading COCO 2017 val images..."
wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip && rm val2017.zip

# Step 2: Download annotations
echo "Downloading annotations..."
wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip

# Step 3: Extract 5k prompts and images using Python
python3 <<EOF
import json, random, pathlib, shutil
from tqdm import tqdm

# Load captions
with open("annotations/captions_val2017.json", "r") as f:
    captions_data = json.load(f)

annotations = captions_data["annotations"]
images_map = {img["id"]: img["file_name"] for img in captions_data["images"]}

# Select 5k random captions
random.seed(42)
sampled = random.sample(annotations, 5000)

# Prepare directories
pathlib.Path("coco_5k_prompts.txt").unlink(missing_ok=True)
out_dir = pathlib.Path("coco_5k_images")
out_dir.mkdir(exist_ok=True)

# Write prompts and copy corresponding images
with open("coco_5k_prompts.txt", "w") as f:
    for idx, ann in enumerate(tqdm(sampled, desc="Processing 5k prompts"), start=1):
        prompt = ann["caption"].strip()
        f.write(f"{prompt}\n")

        image_id = ann["image_id"]
        filename = images_map[image_id]
        src_path = pathlib.Path("val2017") / filename
        dst_path = out_dir / f"{idx:05d}.jpg"
        shutil.copy(src_path, dst_path)
EOF

echo "✅ Done: 5k prompts saved to data/coco2017/coco_5k_prompts.txt"
echo "✅ Images saved to data/coco2017/coco_5k_images/"
