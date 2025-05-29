from diffusers import StableDiffusionPipeline
import torch
import os
from tqdm import tqdm

model_id = "runwayml/stable-diffusion-v1-5"
prompts_file = "data/coco/coco_5k_prompts.txt"
out_dir = "results/gen/sd15_coco/runwayml_stable-diffusion-v1-5_cfg7.5"
cfg_scale = 7.5
start_index = 4179
end_index = 5000

os.makedirs(out_dir, exist_ok=True)

with open(prompts_file, "r") as f:
    prompts = [line.strip() for line in f.readlines()]

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

for i in tqdm(range(start_index, end_index), desc="Generating images"):
    prompt = prompts[i]
    image = pipe(prompt, guidance_scale=cfg_scale).images[0]
    image_path = os.path.join(out_dir, f"{i:05}.png")
    image.save(image_path)
