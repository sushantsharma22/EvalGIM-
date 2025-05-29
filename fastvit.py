from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

image = Image.open("/home/sharm2s1/EvalGIM-/results/gen/full_run_5k/ImageInception_ArtifyAI-v1.0_cfg7.5/00001.png").convert("RGB")

processor = AutoImageProcessor.from_pretrained("nvidia/FasterViT-M")
model = AutoModelForImageClassification.from_pretrained("nvidia/FasterViT-M")

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class = logits.argmax(-1).item()
print(f"Predicted class ID: {predicted_class}")
