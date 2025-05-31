import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
from pathlib import Path

model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
prompts = [
    "a futuristic cityscape at night",
    "a realistic portrait of a woman in natural light"
]
output_dir = Path("results/test_realistic_vision")
output_dir.mkdir(parents=True, exist_ok=True)

print("[INFO] Loading model...")

# Load pipeline base first
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None
)

# Replace scheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.set_progress_bar_config(disable=True)

if not callable(pipe):
    print("[ERROR] Pipeline is not callable. Something is misconfigured.")
    exit(1)

print("[INFO] Generating images...")
try:
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        result = pipe(prompts, guidance_scale=7.5)

    for i, img in enumerate(result.images):
        out_path = output_dir / f"test_{i:02d}.png"
        img.save(out_path)
        print(f"[INFO] Saved: {out_path} ({img.size[0]}x{img.size[1]})")

except Exception as e:
    print(f"[ERROR] Generation failed: {e}")
