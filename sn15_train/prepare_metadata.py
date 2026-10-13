#!/usr/bin/env python3
"""Build DiffSynth-Studio metadata CSVs from <stem>.png + <stem>.txt pairs.

- prompt = original caption (no trigger word; the LoRA is only ever used for this one style).
- All pairs are used for training: the goal is style + prompt fidelity, not reproducing a held-out
  ground truth, so evaluation uses newly written prompts instead (see validate.py).
- max_timestep_boundary column caps training to sigma >= ~0.25, the range the 8-step sampler
  visits (its sigmas are 1.0 ... 0.3). Pass it with --extra_inputs max_timestep_boundary.

Outputs (in --out_dir): train.csv (all pairs), smoke.csv (first --smoke_rows pairs)
"""
import argparse, csv, glob, os, re


def read_pairs(data_dir):
    rows = []
    for png in sorted(glob.glob(os.path.join(glob.escape(data_dir), "*.png"))):
        stem = os.path.splitext(png)[0]
        with open(stem + ".txt", encoding="utf-8-sig") as f:
            caption = re.sub(r"\s+", " ", f.read()).strip()
        if not caption:
            raise ValueError(f"empty caption: {stem}.txt")
        rows.append({"image": os.path.basename(png), "prompt": caption})
    return rows


def write_csv(path, rows, boundary):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "prompt", "max_timestep_boundary"],
                           quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "max_timestep_boundary": boundary})


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=os.path.join(here, "data"), help="folder with png/txt pairs (ASCII symlink)")
    p.add_argument("--out_dir", default=os.path.join(here, "meta"))
    p.add_argument("--max_timestep_boundary", type=float, default=0.9)
    p.add_argument("--smoke_rows", type=int, default=20)
    args = p.parse_args()

    rows = read_pairs(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "train.csv"), rows, args.max_timestep_boundary)
    write_csv(os.path.join(args.out_dir, "smoke.csv"), rows[:args.smoke_rows], args.max_timestep_boundary)
    print(f"pairs={len(rows)} -> {args.out_dir}/train.csv, smoke.csv")


if __name__ == "__main__":
    main()
