#!/usr/bin/env bash
set -e
mkdir -p ../data/coco

# images (val2014 ≈ 6.3 GB)
wget -O val2014.zip  "http://images.cocodataset.org/zips/val2014.zip"
unzip val2014.zip -d ../data/coco && rm val2014.zip

# captions annotations (≈ 240 KB)
wget -O captions_val2014.json \
  "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
unzip annotations_trainval2014.zip annotations/captions_val2014.json -d ../data/coco \
  && rm annotations_trainval2014.zip
