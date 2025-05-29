from diffusers import StableDiffusionPipeline
import torch
import os
from tqdm import tqdm

# Settings
model_id = "stabilityai/stable-diffusion-2-1"
prompts_file = "data/coco/coco_5k_prompts.txt"
out_dir = "results/gen/sd21_coco/stabilityai_stable-diffusion-2-1_cfg7.5"
cfg_scale = 7.5
start_index = 1036  # Already generated
end_index = 5000

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# Load prompts
with open(prompts_file, "r") as f:
    prompts = [line.strip() for line in f.readlines()]

# Load model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Override safety checker to avoid NSFW filter issues
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# Generate remaining images
for i in tqdm(range(start_index, end_index), desc="Generating images"):
    prompt = prompts[i]
    image = pipe(prompt, guidance_scale=cfg_scale).images[0]
    image_path = os.path.join(out_dir, f"{i:05}.png")
    image.save(image_path)
