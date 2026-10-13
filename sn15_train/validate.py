#!/usr/bin/env python3
"""8-step comparison grid: control vs A vs B style LoRAs (run after training, never alongside it).

Configs (grid columns):
  control       base + 8-step V2                        8 steps, cfg 1.0
  A@N           base + 8-step V2 + A step-N checkpoint   8 steps, cfg 1.0
  B@N           base + 8-step V2 + B step-N checkpoint   8 steps, cfg 1.0
  B@N/50step    base + B step-N (no 8-step LoRA)         50 steps, cfg 4.0   (only via --ref50)
Grid rows: prompts x seeds. Each image is cached under <out_dir>/imgs/<config>/, so a rerun only
renders what is missing (e.g. after adding checkpoints). LoRAs are hot-loaded so they can be swapped.

Usage:
  /root/venvs/sn15/bin/python validate.py                    # every 2000 steps + last, A and B
  /root/venvs/sn15/bin/python validate.py --steps 1500,2000  # specific checkpoints
  /root/venvs/sn15/bin/python validate.py --ref50 B:2000     # add a 50-step reference column
  /root/venvs/sn15/bin/python validate.py --dry_run          # print the plan, no GPU
"""
import argparse, csv, glob, json, os, re, time

ROOT = "/root/source/test_SenseNova-U1.5"
DS = f"{ROOT}/DiffSynth-Studio"
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", f"{DS}/models")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
LORA8 = os.path.join(os.environ["DIFFSYNTH_MODEL_BASE_PATH"],
                     "SenseNova/SenseNova-U1.5-8B-MoT-LoRAs/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors")

# Outfits not in the dataset, written in the same tag style as the training captions.
NOVEL = [
    "classic maid dress with frilled apron and puff sleeves, black, white, and lavender, lace trim and ribbon details, glossy satin and crisp cotton, frilled headdress with ribbon, cameo brooch, white thigh-high stockings, strapped mary jane shoes, silver tea tray, maid cafe theme, elegant and cute aesthetic",
    "cropped techwear hoodie with pleated mini skirt, neon cyan, hot pink, and black, holographic circuit pattern, glossy vinyl and mesh fabric, cat-ear headphones with glowing lights, fingerless gloves, chunky platform sneakers, floating holographic screens, cyberpunk street theme, cool and energetic aesthetic",
    "traditional furisode kimono with wide obi sash and long flowing sleeves, deep indigo, crimson, and gold, crane and cherry blossom pattern, silk brocade with embroidered details, ornate kanzashi hair ornaments with dangling flowers, tassel charm, wooden geta sandals, folding fan, new year festival theme, graceful and elegant aesthetic",
    "knight armor outfit with tabard and short cape, silver, royal blue, and gold, heraldic lion emblem, polished plate metal and velvet fabric, winged circlet, gauntlets, armored boots, longsword and round shield, medieval knight prince theme, brave and noble aesthetic",
    "fitted qipao dress with mandarin collar and side slits, jade green, ivory, and gold, peony and cloud embroidery, smooth silk brocade with frog button closures, twin bun hair with tasseled hairpins, jade bangle, embroidered flats, round paper fan, lunar new year theme, elegant and charming aesthetic",
    "wizard academy uniform with hooded cloak and pleated skirt, midnight navy, burgundy, and brass, constellation pattern, wool fabric with velvet lining, pointed witch hat with star charm, crest brooch, striped knee socks, buckled loafers, floating spellbook and wand, magic school theme, mysterious and cute aesthetic",
    "pastel bunny overall dress with puffy blouse, baby pink, cream, and mint, polka dot and carrot pattern, soft corduroy and fluffy faux fur, floppy bunny-ear hood, bunny tail pompom, frilly ankle socks, round-toe strap shoes, basket of painted eggs, spring easter theme, sweet and playful aesthetic",
    "butler tailcoat suit with brocade waistcoat and ruffled cravat, jet black, white, and antique gold, damask pattern, crisp wool and satin, white gloves, pocket watch chain, monocle, polished oxford shoes, silver candelabra, mansion butler theme, refined and gentlemanly aesthetic",
]
# Prompt forms the captions never use: is hair/eye colour controllable, and do short or plain prompts still get the style?
PROBES = [
    "girl with short green hair and red eyes, sleeveless summer sundress with wide straw hat, mint green and white, small daisy pattern, light cotton fabric, ribbon sandals, beach holiday theme, fresh and cheerful aesthetic",
    "a cute boy in a yellow raincoat holding a clear umbrella",
    "red hoodie, denim shorts, sneakers",
]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_prompts(meta_dir, n_train):
    prompts = [{"id": f"n{i+1:02d}", "kind": "novel outfit", "prompt": p} for i, p in enumerate(NOVEL)]
    prompts += [{"id": f"p{i+1:02d}", "kind": "prompt-form probe", "prompt": p} for i, p in enumerate(PROBES)]
    # A few training captions, shown next to their ground truth: style reference + copy (overfit) check.
    train = read_csv(os.path.join(meta_dir, "train.csv"))
    picked = [train[int(i * len(train) / n_train)] for i in range(n_train)]
    prompts += [{"id": f"t{i+1:02d}", "kind": "train caption", "ref_image": r["image"], "prompt": r["prompt"]}
                for i, r in enumerate(picked)]
    return prompts


def list_ckpts(run_dir, every, steps):
    found = {int(re.search(r"step-(\d+)\.safetensors$", f).group(1)): f
             for f in glob.glob(os.path.join(run_dir, "step-*.safetensors"))}
    if not found:
        return []
    if steps:
        keep = [s for s in steps if s in found]
    else:
        keep = [s for s in sorted(found) if s % every == 0]
        if max(found) not in keep:
            keep.append(max(found))
    return [(s, found[s]) for s in keep]


def build_configs(args):
    configs = [] if args.no_control else [{"name": "control", "use8": True, "ckpt": None, "steps": 8, "cfg": 1.0}]
    steps = [int(s) for s in args.steps.split(",")] if args.steps else None
    for item in args.runs.split(","):
        tag, run_dir = item.split("=", 1)
        ckpts = list_ckpts(run_dir, args.every, steps)
        if not ckpts:
            print(f"[warn] no step-*.safetensors in {run_dir}")
        configs += [{"name": f"{tag}@{s}", "use8": True, "ckpt": path, "steps": 8, "cfg": 1.0} for s, path in ckpts]
        for ref in filter(None, (args.ref50 or "").split(",")):
            ref_tag, ref_step = ref.split(":")
            if ref_tag == tag:
                path = os.path.join(run_dir, f"step-{ref_step}.safetensors")
                configs.append({"name": f"{tag}@{ref_step}_50step", "use8": False, "ckpt": path, "steps": 50, "cfg": 4.0})
    return configs


def make_grids(out_dir, prompts, configs, seeds, thumb):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=max(14, thumb // 14))
    except TypeError:
        font = ImageFont.load_default()
    head = thumb // 8
    os.makedirs(os.path.join(out_dir, "grids"), exist_ok=True)

    def grid(rows, cols, cell, col_labels, row_labels, path):
        side = thumb // 2
        canvas = Image.new("RGB", (side + thumb * len(cols), head + thumb * len(rows)), "white")
        draw = ImageDraw.Draw(canvas)
        for j, label in enumerate(col_labels):
            draw.text((side + j * thumb + 4, 4), label, fill="black", font=font)
        for i, label in enumerate(row_labels):
            draw.text((4, head + i * thumb + 4), label, fill="black", font=font)
            for j in range(len(cols)):
                p = cell(rows[i], cols[j])
                if p and os.path.exists(p):
                    canvas.paste(Image.open(p).convert("RGB").resize((thumb, thumb)), (side + j * thumb, head + i * thumb))
        canvas.save(path, quality=90)

    img = lambda cfg, pid, seed: os.path.join(out_dir, "imgs", cfg["name"], f"{pid}_s{seed}.png")
    names = [c["name"] for c in configs]
    # One overview per seed: rows = prompts, cols = configs.
    for seed in seeds:
        grid(prompts, configs, lambda p, c: img(c, p["id"], seed), names, [p["id"] for p in prompts],
             os.path.join(out_dir, "grids", f"overview_s{seed}.jpg"))
    # One grid per prompt: rows = seeds, cols = configs (+ the ground-truth image when there is one).
    for p in prompts:
        cols = list(configs)
        labels = list(names)
        if p.get("ref_image"):
            cols = [{"name": "GT"}] + cols
            labels = ["GT"] + labels
        cell = lambda seed, c: (os.path.join(HERE, "data", p["ref_image"]) if c["name"] == "GT" else img(c, p["id"], seed))
        grid(seeds, cols, cell, labels, [f"s{s}" for s in seeds], os.path.join(out_dir, "grids", f"{p['id']}.jpg"))
    print(f"grids -> {os.path.join(out_dir, 'grids')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=f"A={DS}/models/train/sn15_style_A,B={DS}/models/train/sn15_style_B")
    ap.add_argument("--every", type=int, default=2000, help="evaluate step-N where N %% every == 0, plus the last one")
    ap.add_argument("--steps", default=None, help="explicit comma list of steps (overrides --every)")
    ap.add_argument("--seeds", default="42,1234")
    ap.add_argument("--ref50", default=None, help="e.g. B:2000,A:2000 -> extra 50-step cfg-4 column without the 8-step LoRA")
    ap.add_argument("--no_control", action="store_true")
    ap.add_argument("--n_train", type=int, default=3, help="number of training captions to include (with GT)")
    ap.add_argument("--meta_dir", default=os.path.join(HERE, "meta"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "eval"))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--thumb", type=int, default=256)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    prompts = build_prompts(args.meta_dir, args.n_train)
    configs = build_configs(args)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump({"prompts": prompts, "configs": configs, "seeds": seeds},
              open(os.path.join(args.out_dir, "plan.json"), "w"), indent=1, ensure_ascii=False)
    todo = [(c, p, s) for c in configs for p in prompts for s in seeds
            if not os.path.exists(os.path.join(args.out_dir, "imgs", c["name"], f"{p['id']}_s{s}.png"))]
    print(f"{len(prompts)} prompts x {len(seeds)} seeds x {len(configs)} configs; {len(todo)} images to render")
    for c in configs:
        print(f"  {c['name']:<18} steps={c['steps']} cfg={c['cfg']} 8step={c['use8']} ckpt={c['ckpt']}")
    if args.dry_run:
        return

    import torch
    from diffsynth.core import load_state_dict
    from diffsynth.pipelines.sensenova_u1_image import SenseNovaU1ImagePipeline, ModelConfig

    pipe = SenseNovaU1ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="model*.safetensors")],
        tokenizer_config=ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="./"),
    )
    pipe.dit = pipe.enable_lora_hot_loading(pipe.dit)  # unfused LoRAs, swappable via clear_lora()
    lora8 = load_state_dict(LORA8, torch_dtype=torch.bfloat16, device="cuda")

    for c in configs:
        jobs = [(p, s) for (cc, p, s) in todo if cc is c]
        if not jobs:
            continue
        print(f"== {c['name']} ({len(jobs)} images)")
        pipe.clear_lora(verbose=0)
        if c["use8"]:
            pipe.load_lora(pipe.dit, state_dict=lora8)  # expect "294 tensors are patched"
        if c["ckpt"]:
            pipe.load_lora(pipe.dit, c["ckpt"])         # expect "294 tensors are patched"
        os.makedirs(os.path.join(args.out_dir, "imgs", c["name"]), exist_ok=True)
        for p, s in jobs:
            t = time.time()
            image = pipe(prompt=p["prompt"], seed=s, height=args.size, width=args.size,
                         num_inference_steps=c["steps"], cfg_scale=c["cfg"], shift=3.0,
                         progress_bar_cmd=lambda x: x)
            image.save(os.path.join(args.out_dir, "imgs", c["name"], f"{p['id']}_s{s}.png"))
            print(f"  {p['id']} s{s} {time.time() - t:.1f}s")

    make_grids(args.out_dir, prompts, configs, seeds, args.thumb)


if __name__ == "__main__":
    main()
