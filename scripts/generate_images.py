import argparse, json, torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
import os

def safe_model_name(model_id):
    return model_id.replace("/", "_").replace("-", "_")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    # Define subfolder using model ID and cfg
    model_name = safe_model_name(args.model_id)
    subfolder = f"{model_name}_cfg{args.cfg_scale}"
    out_path = Path(args.out_dir) / subfolder
    out_path.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = Path(args.prompts_file).read_text().splitlines()
    existing_files = {p.stem for p in out_path.glob("*.png")}

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe.to(device)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    pipe.set_progress_bar_config(disable=True)

    remaining = []
    for idx, prompt in enumerate(prompts):
        filename = f"{idx:05d}.png"
        if filename[:-4] not in existing_files:
            remaining.append((idx, prompt))

    accelerator.print(f"[INFO] {len(existing_files)} exist, {len(remaining)} to generate")

    for i in tqdm(range(0, len(remaining), args.batch_size)):
        batch = remaining[i:i + args.batch_size]
        indices, texts = zip(*batch)

        try:
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                results = pipe(list(texts), num_inference_steps=30, guidance_scale=args.cfg_scale)
            images = results.images
        except Exception as e:
            accelerator.print(f"[ERROR] Failed batch {i}: {e}")
            continue

        for img, idx in zip(images, indices):
            fname = out_path / f"{idx:05d}.png"
            try:
                img.save(fname)
            except Exception as e:
                accelerator.print(f"[ERROR] Failed to save image {fname.name}: {e}")

    json.dump({
        "model": args.model_id,
        "cfg_scale": args.cfg_scale,
        "total_prompts": len(prompts)
    }, open(out_path / "meta.json", "w"), indent=2)

    accelerator.print(f"[DONE] Images saved to {out_path}")

if __name__ == "__main__":
    main()
