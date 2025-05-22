import argparse, json, torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True)
    p.add_argument("--prompts_file", required=True)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--out_dir", default="results/gen")
    p.add_argument("--failed_log", default="failed_prompts.txt")
    args = p.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe.to(device)

    # Patch: Disable NSFW checker safely
    if pipe.safety_checker is not None:
        def null_checker(images, **kwargs):
            return images, [False] * len(images)
        pipe.safety_checker = null_checker

    pipe.set_progress_bar_config(disable=True)

    prompts = Path(args.prompts_file).read_text().splitlines()
    out_root = Path(args.out_dir) / f"{args.model_id.replace('/','_')}_cfg{args.cfg_scale}"
    out_root.mkdir(parents=True, exist_ok=True)
    failed_log = Path(args.failed_log)

    with failed_log.open("a") as flog:
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            batch = prompts[i : i + args.batch_size]
            try:
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    result = pipe(batch, num_inference_steps=30, guidance_scale=args.cfg_scale)

                images = getattr(result, "images", None)
                if not isinstance(images, list):
                    raise ValueError("Pipeline did not return valid list of images.")

            except Exception as e:
                accelerator.print(f"[WARNING] Batch {i} failed: {e}")
                for j, prompt in enumerate(batch):
                    flog.write(f"{i + j}: {prompt}\n")
                continue

            for j, (prompt, img) in enumerate(zip(batch, images)):
                img_idx = i + j
                fname = out_root / f"{img_idx:05d}.png"
                if fname.exists():
                    continue
                try:
                    img.save(fname)
                    accelerator.print(f"[SAVED] {fname}")
                except Exception as e:
                    accelerator.print(f"[ERROR] Saving failed for {fname.name}: {e}")
                    flog.write(f"{img_idx}: {prompt}\n")

    meta = {"model": args.model_id, "cfg": args.cfg_scale, "count": len(prompts)}
    json.dump(meta, open(out_root / "meta.json", "w"), indent=2)
    accelerator.print(f"✅ Done! Images in {out_root}")

if __name__ == "__main__":
    main()
