#!/usr/bin/env python3
"""A-vs-B progress grids from the checkpoint-time samples written by train_sn15.py (CPU only, safe to
run while training).

For each prompt: rows = runs (A, B), columns = step-0 (no style), step-500, ... ; training-caption
prompts get a GT column. Output: <out_dir>/<prompt id>.jpg and <out_dir>/all_<run>.jpg (rows = prompts).

  /root/venvs/sn15/bin/python compare_samples.py               # every saved step
  /root/venvs/sn15/bin/python compare_samples.py --every 1000  # thinner columns
"""
import argparse, glob, json, os, re
from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/source/test_SenseNova-U1.5"
HERE = os.path.dirname(os.path.abspath(__file__))


def steps_of(sample_dir):
    return sorted(int(m.group(1)) for d in glob.glob(os.path.join(sample_dir, "step-*"))
                  if os.path.isdir(d) and (m := re.search(r"step-(\d+)$", d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=f"A={ROOT}/DiffSynth-Studio/models/train/sn15_style_A,"
                                      f"B={ROOT}/DiffSynth-Studio/models/train/sn15_style_B")
    ap.add_argument("--every", type=int, default=500, help="keep step-0, multiples of this, and the last step")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "eval", "progress"))
    ap.add_argument("--thumb", type=int, default=256)
    args = ap.parse_args()

    runs = [(tag, os.path.join(path, "samples")) for tag, path in (r.split("=", 1) for r in args.runs.split(","))]
    runs = [(tag, d) for tag, d in runs if os.path.isdir(d)]
    if not runs:
        raise SystemExit("no samples/ directories found")
    prompts = json.load(open(os.path.join(runs[0][1], "prompts.json")))["prompts"]
    all_steps = sorted(set().union(*(steps_of(d) for _, d in runs)))
    last = max(all_steps)
    steps = [s for s in all_steps if s % args.every == 0 or s == last]

    t, head, side = args.thumb, 28, 60
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    def grid(row_labels, col_labels, cell, path):
        canvas = Image.new("RGB", (side + t * len(col_labels), head + t * len(row_labels)), "white")
        draw = ImageDraw.Draw(canvas)
        for j, label in enumerate(col_labels):
            draw.text((side + j * t + 4, 6), label, fill="black", font=font)
        for i, label in enumerate(row_labels):
            draw.text((4, head + i * t + 4), label, fill="black", font=font)
            for j in range(len(col_labels)):
                p = cell(i, j)
                if p and os.path.exists(p):
                    canvas.paste(Image.open(p).convert("RGB").resize((t, t)), (side + j * t, head + i * t))
        canvas.save(path, quality=88)

    os.makedirs(args.out_dir, exist_ok=True)
    cols = [f"step-{s}" for s in steps]
    for p in prompts:
        gt = os.path.join(HERE, "data", p["ref_image"]) if p.get("ref_image") else None
        labels = (["GT"] if gt else []) + cols
        def cell(i, j, p=p, gt=gt):
            if gt:
                if j == 0:
                    return gt
                j -= 1
            return os.path.join(runs[i][1], cols[j], f"{p['id']}.png")
        grid([tag for tag, _ in runs], labels, cell, os.path.join(args.out_dir, f"{p['id']}.jpg"))
    for tag, d in runs:
        grid([p["id"] for p in prompts], cols, lambda i, j, d=d: os.path.join(d, cols[j], f"{prompts[i]['id']}.png"),
             os.path.join(args.out_dir, f"all_{tag}.jpg"))
    print(f"{len(prompts)} prompts x {len(steps)} steps x {len(runs)} runs -> {args.out_dir}")


if __name__ == "__main__":
    main()
