import argparse, json, torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    # Prepare output folder
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = Path(args.prompts_file).read_text().splitlines()
    end = args.end if args.end is not None else len(prompts)
    prompts = prompts[args.start:end]
    offset = args.start

    # Check existing images
    existing = {p.stem for p in out_path.glob("*.png")}
    remaining = [(i + offset, prompt) for i, prompt in enumerate(prompts)
                 if f"{i + offset:05d}" not in existing]

    accelerator.print(f"[INFO] {len(existing)} exist, {len(remaining)} to generate")

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe.to(device)
    if pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    pipe.set_progress_bar_config(disable=True)

    # Generate in batches
    for i in range(0, len(remaining), args.batch_size):
        batch = remaining[i:i + args.batch_size]
        indices, texts = zip(*batch)

        try:
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                results = pipe(list(texts), guidance_scale=args.cfg_scale)
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

    # Save metadata
    json.dump({
        "model": args.model_id,
        "cfg_scale": args.cfg_scale,
        "total_prompts": len(prompts),
        "start": args.start,
        "end": end
    }, open(out_path / "meta.json", "w"), indent=2)

    accelerator.print(f"[DONE] Images saved to {out_path}")

if __name__ == "__main__":
    main()
