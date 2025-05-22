import json, random, pathlib, argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--captions_json", required=True)
parser.add_argument("--n", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="coco_5k_prompts.txt")
args = parser.parse_args()

random.seed(args.seed)

with open(args.captions_json, encoding='utf-8', errors='replace') as f:
    ann = json.load(f)

captions = [c["caption"].strip() for img in ann["annotations"] for c in [img]]
prompts = random.sample(captions, args.n)

pathlib.Path(args.out).write_text("\n".join(prompts))
print(f"Saved {args.n} prompts to {args.out}")
