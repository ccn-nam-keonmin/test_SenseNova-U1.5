#!/usr/bin/env python3
"""DiffSynth's SenseNova-U1 train.py + 8-step sample images at every checkpoint.

Training is unchanged: this loads the training module and parser from
DiffSynth-Studio/examples/sensenova_u1/model_training/train.py and repeats its __main__ block,
swapping ModelLogger for SamplingModelLogger. Right after each checkpoint is saved (and once
before training, as the no-style "step-0" reference) it renders a fixed prompt set with the
weights being trained, in the same process (no second model copy on the GPU).

  A (8-step LoRA fused via --preset_lora_path): sampled as is.
  B (plain base): pass --sample_lora8 <8-step file>. It is applied as a temporary forward hook
    on its 294 target modules while sampling and removed afterwards; base weights are untouched.

Output: <output_path>/samples/step-N/<prompt id>.png and a contact sheet <output_path>/samples/step-N.jpg
"""
import importlib.util, json, os, sys, time

import accelerate
import torch
from PIL import Image, ImageDraw, ImageFont

DS = "/root/source/test_SenseNova-U1.5/DiffSynth-Studio"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_spec = importlib.util.spec_from_file_location(
    "sensenova_official_train", os.path.join(DS, "examples/sensenova_u1/model_training/train.py"))
official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(official)

from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.diffusion import ModelLogger, launch_training_task
from diffsynth.models.sensenova_u1_common import PATCH_SIZE
from validate import build_prompts


class Sampler:
    def __init__(self, prompts, out_dir, seed=42, steps=8, cfg=1.0, size=1024, lora8_path=None, thumb=256):
        self.prompts, self.out_dir = prompts, out_dir
        self.seed, self.steps, self.cfg, self.size = seed, steps, cfg, size
        self.lora8_path, self.lora8, self.thumb = lora8_path, None, thumb
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "prompts.json"), "w") as f:
            json.dump({"prompts": prompts, "seed": seed, "steps": steps, "cfg": cfg, "size": size,
                       "lora8": lora8_path}, f, indent=1, ensure_ascii=False)

    def _lora8_hooks(self, pipe):
        if self.lora8_path is None:
            return []
        if self.lora8 is None:
            sd = load_state_dict(self.lora8_path, torch_dtype=pipe.torch_dtype, device=pipe.device)
            sd = pipe.lora_loader(torch_dtype=pipe.torch_dtype, device=pipe.device).convert_state_dict(sd)
            names = sorted(k[:-len(".lora_A.weight")] for k in sd if k.endswith(".lora_A.weight"))
            self.lora8 = [(n, sd[n + ".lora_A.weight"], sd[n + ".lora_B.weight"]) for n in names]
        linear = torch.nn.functional.linear
        return [pipe.dit.get_submodule(name).register_forward_hook(
                    lambda m, inp, out, A=A, B=B: out + linear(linear(inp[0], A), B))
                for name, A, B in self.lora8]

    def __call__(self, pipe, tag):
        t0 = time.time()
        out = os.path.join(self.out_dir, tag)
        os.makedirs(out, exist_ok=True)
        hooks = self._lora8_hooks(pipe)
        try:
            for p in self.prompts:
                image = pipe(prompt=p["prompt"], seed=self.seed, height=self.size, width=self.size,
                             num_inference_steps=self.steps, cfg_scale=self.cfg, shift=3.0,
                             rand_device=pipe.device, progress_bar_cmd=lambda x: x)
                image.save(os.path.join(out, f"{p['id']}.png"))
        finally:
            for h in hooks:
                h.remove()
            # pipe() re-sets the scheduler for inference; restore what switch_pipe_to_training_mode set.
            pipe.scheduler.set_timesteps(1000, training=True)
        self._contact_sheet(out, tag)
        print(f"[samples] {tag}: {len(self.prompts)} images in {time.time() - t0:.0f}s -> {out}", flush=True)

    def _contact_sheet(self, out, tag, cols=7):
        t = self.thumb
        rows = (len(self.prompts) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * t, rows * (t + 22) + 30), "white")
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.load_default(size=16)
        except TypeError:
            font = ImageFont.load_default()
        draw.text((6, 6), tag, fill="black", font=font)
        for i, p in enumerate(self.prompts):
            x, y = (i % cols) * t, 30 + (i // cols) * (t + 22)
            sheet.paste(Image.open(os.path.join(out, f"{p['id']}.png")).convert("RGB").resize((t, t)), (x, y + 22))
            draw.text((x + 4, y + 2), p["id"], fill="black", font=font)
        sheet.save(os.path.join(self.out_dir, f"{tag}.jpg"), quality=90)


class SamplingModelLogger(ModelLogger):
    def __init__(self, *args, sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler

    def save_model(self, accelerator, model, file_name):
        super().save_model(accelerator, model, file_name)
        if self.sampler is not None and accelerator.is_main_process:
            self.sampler(accelerator.unwrap_model(model).pipe, os.path.splitext(file_name)[0])


def parser():
    p = official.sensenova_u1_image_parser()
    g = p.add_argument_group("samples at checkpoints")
    g.add_argument("--no_samples", action="store_true", help="Disable checkpoint-time sampling.")
    g.add_argument("--no_sample_at_start", action="store_true", help="Skip the step-0 (no style) samples.")
    g.add_argument("--sample_lora8", type=str, default=None,
                   help="8-step LoRA applied (as a hook) only while sampling. Use for B; never together with --preset_lora_path.")
    g.add_argument("--sample_seed", type=int, default=42)
    g.add_argument("--sample_steps", type=int, default=8)
    g.add_argument("--sample_cfg", type=float, default=1.0)
    g.add_argument("--sample_size", type=int, default=1024)
    g.add_argument("--sample_meta_dir", type=str, default=os.path.join(HERE, "meta"),
                   help="Where train.csv is (for the training-caption prompts).")
    g.add_argument("--sample_n_train", type=int, default=3, help="How many training captions to include.")
    return p


def main():
    args = parser().parse_args()
    if args.sample_lora8 and args.preset_lora_path:
        raise SystemExit("--sample_lora8 with --preset_lora_path would apply the 8-step LoRA twice.")

    # ---- same as examples/sensenova_u1/model_training/train.py __main__ ----
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=PATCH_SIZE,
            width_division_factor=PATCH_SIZE,
        ),
    )
    model = official.SenseNovaU1ImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_config=args.tokenizer_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        template_model_id_or_path=args.template_model_id_or_path,
        enable_lora_hot_loading=args.enable_lora_hot_loading,
    )
    # ---- sampling setup (the only addition) ----
    sampler = None
    if not args.no_samples and accelerator.is_main_process:
        sampler = Sampler(
            build_prompts(args.sample_meta_dir, args.sample_n_train),
            os.path.join(args.output_path, "samples"),
            seed=args.sample_seed, steps=args.sample_steps, cfg=args.sample_cfg,
            size=args.sample_size, lora8_path=args.sample_lora8,
        )
        if not args.no_sample_at_start and not (args.initialize_model_on_cpu or args.enable_model_cpu_offload):
            sampler(model.pipe, "step-0")
    model_logger = SamplingModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        enable_csv_log=args.enable_csv_log,
        sampler=sampler,
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
