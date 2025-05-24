#!/usr/bin/env python
import argparse, json, torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from PIL import Image
from pathlib import Path

def load_prompts(path):
    return Path(path).read_text().splitlines()

def parse_failed(log_path):
    if not log_path.exists():
        return []
    failed = []
    for line in log_path.read_text().splitlines():
        if ": " in line:
            idx, prompt = line.split(": ", 1)
            try:
                failed.append((int(idx), prompt))
            except:
                pass
    return failed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",     required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--cfg_scale",    type=float, default=7.5)
    parser.add_argument("--batch_size",   type=int,   default=1)
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--failed_log",   default="failed_prompts.txt")
    args = parser.parse_args()

    # 1) Load model
    accel = Accelerator()
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(accel.device)
    pipe.safety_checker = lambda images, **kw: (images, False)
    pipe.set_progress_bar_config(disable=True)

    # 2) Load prompts & figure out work
    prompts = load_prompts(args.prompts_file)
    total   = len(prompts)

    # 3) Use out_dir exactly as passed
    out_root = Path(args.out_dir)
    print(f"[DEBUG] Writing to OUT_DIR = {args.out_dir}", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)

    existing = {int(p.stem) for p in out_root.glob("*.png")}
    to_do    = [i for i in range(total) if i not in existing]
    print(f"[INFO] {len(existing)} exist, {len(to_do)} to generate", flush=True)

    # 4) Clear old failures
    flog = Path(args.failed_log)
    if flog.exists(): flog.unlink()

    # 5) Generate missing
    with flog.open("a") as f:
        pbar = tqdm(to_do, total=len(to_do), unit="img", ncols=80)
        for idx in pbar:
            pbar.set_description(f"{idx:05d}.png")
            prompt = prompts[idx]
            try:
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    img = pipe([prompt], num_inference_steps=30, guidance_scale=args.cfg_scale).images[0]
                if not isinstance(img, Image.Image):
                    raise ValueError
                img.save(out_root / f"{idx:05d}.png")
                print(f"Saved {idx:05d}.png", flush=True)
            except Exception:
                f.write(f"{idx}: {prompt}\n")
            pbar.update(1)
        pbar.close()

    # 6) Retry failures once
    failed = parse_failed(flog)
    if failed:
        print(f"[INFO] Retrying {len(failed)} failures…", flush=True)
        with flog.open("w") as f:
            for idx, prompt in failed:
                path = out_root / f"{idx:05d}.png"
                if path.exists(): continue
                try:
                    img = pipe([prompt], num_inference_steps=30, guidance_scale=args.cfg_scale).images[0]
                    img.save(path)
                    print(f"Saved {idx:05d}.png", flush=True)
                except:
                    f.write(f"{idx}: {prompt}\n")

    # 7) Write meta
    meta = {"model": args.model_id, "cfg_scale": args.cfg_scale, "total_prompts": total}
    (out_root/"meta.json").write_text(json.dumps(meta, indent=2))
    print("✅ Done!", flush=True)

if __name__=="__main__":
    main()
