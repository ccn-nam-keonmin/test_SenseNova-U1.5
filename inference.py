# /root/venvs/sn15/bin/python inference.py 
"""SenseNova-U1.5 + trained style LoRA (+ optionally the 8-step LoRA V2) -> image(s) in the trained style.

  STEP8 = True : base + 8-step LoRA + style LoRA, 8 steps / cfg 1.0 (fast)
  STEP8 = False: base + style LoRA, 50 steps / cfg 4.0 (the base model's standard settings)

Run:  /root/venvs/sn15/bin/python /root/source/test_SenseNova-U1.5/inference.py
Prompts work best in the same comma-separated tag form as the training captions
(outfit, colors, pattern, material, accessories, theme, mood).
"""
import os
import time

PROMPT = "Pokekoro dress-up game style girl character, sweet cheerful amusement park themed outfit. Wearing a pastel lavender oversized varsity jacket with white striped cuffs, a letter K patch and a small pink bunny patch, layered over a simple white high-neck short sleeve top. Paired with a high-waisted soft yellow plaid mini pleated skirt, light and cute silhouette. The color palette is pastel purple, creamy yellow, white and soft pink accents. Yellow high-top sneakers with purple bow shoelaces and yellow bow ankle details. Holding a cute bunny ice cream cone, with a large yellow bunny balloon and a bunch of pastel colorful balloons decorated with pink bows. Bright, dreamy, sweet, youthful and adorable, perfect for a girly dress-up game character."
STEP8 = False            # True: + 8-step LoRA (8 steps, cfg 1.0) / False: base + style LoRA only (50 steps, cfg 4.0)

# ---- settings ----
STYLE_RUN = "B"          # B: trained on the base model (A breaks the 8-step LoRA; see sn15_train/eval/ab_summary)
STYLE_STEP = 10080       # checkpoint step: 500, 1000, ..., 10000, 10080
STYLE_ALPHA = 1.0        # style strength (1.0 = as trained)
SEED = 42
NUM_IMAGES = 1           # seeds SEED, SEED+1, ... (the model is loaded once)
HEIGHT = WIDTH = 1024    # trained at 1024x1024; must be multiples of 32
NUM_STEPS = 8 if STEP8 else 50      # 8-step LoRA: 8 steps / base model: 50 steps
CFG_SCALE = 1.0 if STEP8 else 4.0   # 8-step LoRA is CFG-distilled (cfg 1.0) / base model: cfg 4.0
SHIFT = 3.0

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "DiffSynth-Studio", "models")
LORA_8STEP = os.path.join(MODELS, "SenseNova/SenseNova-U1.5-8B-MoT-LoRAs/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors")
LORA_STYLE = os.path.join(MODELS, f"train/sn15_style_{STYLE_RUN}/step-{STYLE_STEP}.safetensors")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# Use the local weights only (must be set before importing diffsynth).
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", MODELS)
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

import torch
from diffsynth.pipelines.sensenova_u1_image import SenseNovaU1ImagePipeline, ModelConfig


def main():
    for path in ([LORA_8STEP] if STEP8 else []) + [LORA_STYLE]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    pipe = SenseNovaU1ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="model*.safetensors")],
        tokenizer_config=ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="./"),
    )
    # LoRAs are fused into the weights; each should report "294 tensors are fused by LoRA."
    if STEP8:
        pipe.load_lora(pipe.dit, LORA_8STEP)
    pipe.load_lora(pipe.dit, LORA_STYLE, alpha=STYLE_ALPHA)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i in range(NUM_IMAGES):
        seed = SEED + i
        image = pipe(
            prompt=PROMPT, seed=seed, height=HEIGHT, width=WIDTH,
            num_inference_steps=NUM_STEPS, cfg_scale=CFG_SCALE, shift=SHIFT,
        )
        mode = "8step" if STEP8 else f"{NUM_STEPS}step"
        path = os.path.join(OUTPUT_DIR, f"{stamp}_{STYLE_RUN}{STYLE_STEP}_{mode}_s{seed}.png")
        image.save(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
