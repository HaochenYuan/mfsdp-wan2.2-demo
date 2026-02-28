# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Megatron-FSDP Pretrain Script for Wan2.2-TI2V-5B

import argparse
import logging
import math
import os
import random
import sys
from itertools import cycle
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel, WanAttentionBlock
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.modules.vae2_2 import Wan2_2_VAE

logger = logging.getLogger(__name__)

DP_SHARD = "dp_shard"
DP_OUTER = "dp_outer"
CP = "cp"
DP_SHARD_CP = "dp_shard_cp"
TP = "tp"
HSDP = "hsdp"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain Wan2.2 with Megatron-FSDP / PyTorch FSDP2")

    # Model
    parser.add_argument("--model-config", type=str, default="ti2v-5B",
                        choices=list(WAN_CONFIGS.keys()),
                        help="Model config key from WAN_CONFIGS")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint dir (required unless --random-init)")
    parser.add_argument("--output-dir", type=str, default="./output/wan22_pretrain")
    parser.add_argument("--random-init", action="store_true", default=False,
                        help="Skip checkpoint loading, use random weights (for perf tuning)")

    # Architecture overrides (only effective with --random-init)
    parser.add_argument("--override-dim", type=int, default=None,
                        help="Override hidden dim (e.g. 4096 for ~10B)")
    parser.add_argument("--override-ffn-dim", type=int, default=None,
                        help="Override FFN intermediate dim")
    parser.add_argument("--override-num-layers", type=int, default=None,
                        help="Override number of transformer layers")
    parser.add_argument("--override-num-heads", type=int, default=None,
                        help="Override number of attention heads")

    # Training (iteration-based)
    parser.add_argument("--max-steps", type=int, required=True,
                        help="Total training iterations")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Micro batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)

    # Mock data
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Mock dataset size")
    parser.add_argument("--frame-num", type=int, default=17)
    parser.add_argument("--resolution", type=int, nargs=2, default=[480, 832])

    # Megatron-FSDP
    parser.add_argument("--dp-shard-size", type=int, default=None)
    parser.add_argument("--dp-outer-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--zero-dp-strategy", type=str, default="optim_grads_params",
                        choices=["no_shard", "optim", "optim_grads", "optim_grads_params"])
    parser.add_argument("--outer-dp-strategy", type=str, default="no_shard")
    parser.add_argument("--preserve-fp32-weights", action="store_true", default=False)
    parser.add_argument("--grad-reduce-in-fp32", action="store_true", default=False)
    parser.add_argument("--overlap-grad-reduce", action="store_true", default=False)
    parser.add_argument("--overlap-param-gather", action="store_true", default=False)
    parser.add_argument("--fsdp-double-buffer", action="store_true", default=False)
    parser.add_argument("--nccl-ub", action="store_true", default=False,
                        help="Enable NCCL user buffer registration (auto-enables double buffer)")
    parser.add_argument("--disable-symmetric-registration", action="store_true", default=False,
                        help="Force local UB registration instead of symmetric (window)")

    # FSDP backend
    parser.add_argument("--fsdp-backend", type=str, default="megatron",
                        choices=["megatron", "pytorch"],
                        help="FSDP implementation: 'megatron' (Megatron-FSDP) or 'pytorch' (PyTorch FSDP2)")

    # TransformerEngine
    parser.add_argument("--use-te-linear", action="store_true", default=False,
                        help="Replace nn.Linear with TE Linear")
    parser.add_argument("--fuse-wgrad-accumulation", action="store_true", default=False,
                        help="Enable fuse_wgrad_accumulation in TE Linear (requires --use-te-linear, megatron backend only)")

    # Mixed precision
    parser.add_argument("--param-dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"])

    # Misc
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--log-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    # Profiler
    parser.add_argument("--enable-profiler", action="store_true", default=False)
    parser.add_argument("--profiler-start-step", type=int, default=5,
                        help="Step to start profiling (skip warmup)")
    parser.add_argument("--profiler-end-step", type=int, default=8,
                        help="Step to stop profiling")
    parser.add_argument("--profiler-output-dir", type=str,
                        default="./profiler_output")

    return parser.parse_args()


def replace_linear_with_te(model, fuse_wgrad_accumulation=False, alignment=16):
    """
    Replace nn.Linear layers in the model with te.Linear.
    """
    import transformer_engine.pytorch as te

    replaced = 0
    skipped = 0

    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue

            in_f = child.in_features
            out_f = child.out_features
            has_bias = child.bias is not None

            # Check alignment
            if in_f % alignment != 0 or out_f % alignment != 0:
                skipped += 1
                logger.info(f"  Skip {name}.{attr_name}: ({in_f}, {out_f}) "
                            f"not aligned to {alignment}")
                continue

            # Create TE Linear
            te_linear = te.Linear(
                in_f, out_f, bias=has_bias,
                params_dtype=child.weight.dtype,
                fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            )

            # Copy weights
            with torch.no_grad():
                te_linear.weight.copy_(child.weight)
                if has_bias and child.bias is not None:
                    te_linear.bias.copy_(child.bias)

            setattr(module, attr_name, te_linear)
            replaced += 1

    logger.info(f"TE Linear replacement: {replaced} replaced, {skipped} skipped "
                f"(fuse_wgrad_accumulation={fuse_wgrad_accumulation})")
    return model


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")

    return rank, world_size, local_rank


def setup_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def set_seed(seed, rank):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def build_device_mesh(args, world_size):
    if args.dp_shard_size is None:
        args.dp_shard_size = world_size // (args.dp_outer_size * args.cp_size * args.tp_size)

    mesh_shape = (args.dp_outer_size, args.dp_shard_size, args.cp_size, args.tp_size)
    mesh_dim_names = (DP_OUTER, DP_SHARD, CP, TP)

    logger.info(f"Device mesh: {mesh_shape}")

    device_mesh = init_device_mesh("cuda", mesh_shape=mesh_shape,
                                   mesh_dim_names=mesh_dim_names)
    device_mesh[(DP_SHARD, CP)]._flatten(DP_SHARD_CP)
    device_mesh[(DP_OUTER, DP_SHARD, CP)]._flatten(HSDP)

    return device_mesh


class MockDataset(torch.utils.data.Dataset):
    """Mock dataset with random video tensors."""

    def __init__(self, num_samples, frame_num, resolution):
        self.num_samples = num_samples
        self.frame_num = frame_num
        self.resolution = resolution
        self.captions = [
            "A beautiful sunset over the ocean.",
            "A cat playing with yarn.",
            "Time-lapse of clouds over mountains.",
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "video": torch.randn(3, self.frame_num, *self.resolution),
            "text": self.captions[idx % len(self.captions)]
        }


def collate_fn(batch):
    return {
        "video": torch.stack([x["video"] for x in batch]),
        "text": [x["text"] for x in batch]
    }


class FlowMatchingLoss:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps

    def __call__(self, model, x_0, context, seq_len):
        batch_size = len(x_0)
        device = x_0[0].device

        t = torch.rand(batch_size, device=device) * self.num_timesteps
        noise = [torch.randn_like(x) for x in x_0]

        x_t = []
        for i, (x, n) in enumerate(zip(x_0, noise)):
            t_i = (t[i] / self.num_timesteps).view(-1, 1, 1, 1)
            x_t.append((1 - t_i) * x + t_i * n)

        target = [n - x for n, x in zip(noise, x_0)]
        pred = model(x_t, t=t, context=context, seq_len=seq_len)

        return sum(F.mse_loss(p, tgt) for p, tgt in zip(pred, target)) / len(pred)


def get_cosine_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(model, text_encoder, vae, data_iter, optimizer, scheduler, criterion,
          args, rank, device, param_dtype, config):
    """Main training loop with backward timing and profiler support."""

    model.train()
    patch_size = config.patch_size
    grad_accum = args.gradient_accumulation_steps
    use_megatron_fsdp = (args.fsdp_backend == "megatron")

    # ---- Profiler setup ----
    profiler = None
    if args.enable_profiler:
        import torch.profiler as profiler_module

        profiler_dir = Path(args.profiler_output_dir)
        profiler_dir.mkdir(parents=True, exist_ok=True)

        wait_steps = max(0, args.profiler_start_step - 1)
        active_steps = args.profiler_end_step - args.profiler_start_step

        def trace_handler(prof):
            """Export Chrome trace + key_averages table."""
            trace_path = profiler_dir / f"trace_rank{rank}_step{prof.step_num}.json"
            prof.export_chrome_trace(str(trace_path))
            if rank == 0:
                logger.info(f"Chrome trace saved: {trace_path}")
                logger.info(prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=80,
                    max_name_column_width=120))

        profiler = profiler_module.profile(
            activities=[
                profiler_module.ProfilerActivity.CPU,
                profiler_module.ProfilerActivity.CUDA,
            ],
            schedule=profiler_module.schedule(
                wait=wait_steps,
                warmup=1,
                active=active_steps,
                repeat=1,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        )
        profiler.start()
        if rank == 0:
            logger.info(f"Profiler enabled: start_step={args.profiler_start_step}, "
                        f"end_step={args.profiler_end_step}, output={profiler_dir}")

    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    step_start = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)

    # ---- Training loop ----
    for step in range(1, args.max_steps + 1):
        step_start.record()
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        fwd_time_ms = 0.0
        bwd_time_ms = 0.0

        for micro_idx in range(grad_accum):
            is_last = (micro_idx == grad_accum - 1)
            if use_megatron_fsdp:
                model.set_model_auto_sync(is_last)
            else:
                model.set_requires_gradient_sync(is_last)

            if text_encoder is not None and vae is not None:
                batch = next(data_iter)
                videos = torch.randn(args.batch_size, 3, args.frame_num,
                                     *args.resolution, device=device)
                texts = batch["text"]

                with torch.no_grad():
                    context = text_encoder(texts, device)
                    latents = vae.encode([v for v in videos])

                shape = latents[0].shape  # [C, F', H', W']
                f_lat, h_lat, w_lat = shape[1], shape[2], shape[3]
            else:
                vae_stride = config.vae_stride
                in_dim = config.in_dim if hasattr(config, 'in_dim') else 16
                f_lat = args.frame_num // vae_stride[0]
                h_lat = args.resolution[0] // vae_stride[1]
                w_lat = args.resolution[1] // vae_stride[2]
                latents = [torch.randn(in_dim, f_lat, h_lat, w_lat,
                                       device=device, dtype=param_dtype)
                           for _ in range(args.batch_size)]
                context = [torch.randn(config.text_len,
                                       config.text_dim if hasattr(config, 'text_dim') else 4096,
                                       device=device, dtype=param_dtype)
                           for _ in range(args.batch_size)]

            seq_len = (f_lat // patch_size[0]) * \
                      (h_lat // patch_size[1]) * \
                      (w_lat // patch_size[2])

            fwd_start.record()
            with torch.amp.autocast("cuda", dtype=param_dtype):
                loss = criterion(model, latents, context, seq_len) / grad_accum
            fwd_end.record()

            bwd_start.record()
            loss.backward()
            bwd_end.record()

            torch.cuda.synchronize()
            fwd_time_ms += fwd_start.elapsed_time(fwd_end)
            bwd_time_ms += bwd_start.elapsed_time(bwd_end)
            step_loss += loss.item()

        optimizer.step()
        scheduler.step()

        step_end.record()
        torch.cuda.synchronize()
        step_time_ms = step_start.elapsed_time(step_end)

        # Logging
        if rank == 0 and step % args.log_steps == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Step {step}/{args.max_steps} | "
                f"Loss: {step_loss:.4f} | "
                f"Step: {step_time_ms:.2f}ms | "
                f"Forward: {fwd_time_ms:.2f}ms | "
                f"Backward: {bwd_time_ms:.2f}ms | "
                f"LR: {lr:.2e}"
            )

        # Profiler step
        if profiler is not None:
            profiler.step()
            if step >= args.profiler_end_step:
                profiler.stop()
                if rank == 0:
                    logger.info(f"Profiler stopped at step {step}.")
                profiler = None

    if profiler is not None:
        profiler.stop()
        if rank == 0:
            logger.info("Profiler stopped (training ended).")


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    setup_logging(rank)
    set_seed(args.seed, rank)

    logger.info(f"World size: {world_size}, Rank: {rank}")
    logger.info(f"Training for {args.max_steps} iterations")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    param_dtype = dtype_map[args.param_dtype]
    config = WAN_CONFIGS[args.model_config]

    # Validate args
    if not args.random_init and args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required unless --random-init is set")
    if not args.random_init and (args.override_dim or args.override_ffn_dim
                                  or args.override_num_layers or args.override_num_heads):
        raise ValueError("--override-* options require --random-init")

    # Apply architecture overrides
    model_dim = args.override_dim or config.dim
    model_ffn_dim = args.override_ffn_dim or config.ffn_dim
    model_num_layers = args.override_num_layers or config.num_layers
    model_num_heads = args.override_num_heads or config.num_heads

    logger.info(f"Model config: {args.model_config} "
                f"(dim={model_dim}, ffn_dim={model_ffn_dim}, "
                f"layers={model_num_layers}, heads={model_num_heads})"
                f"{' [RANDOM INIT]' if args.random_init else ''}")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.fsdp_backend == "megatron":
        device_mesh = build_device_mesh(args, world_size)
        use_hsdp = args.dp_outer_size > 1
    else:
        device_mesh = None  # FSDP2 creates its own mesh in the wrapping section
        use_hsdp = False

    # ---- Load frozen T5 encoder + VAE (skip if random-init) ----
    text_encoder = None
    vae = None
    if not args.random_init:
        logger.info("Loading T5 encoder...")
        text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=device,
            checkpoint_path=os.path.join(args.checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(args.checkpoint_dir, config.t5_tokenizer),
        )
        text_encoder.model.eval()
        for p in text_encoder.model.parameters():
            p.requires_grad = False

        # ti2v-5B uses Wan2.2_VAE; 14B models use Wan2.1_VAE
        vae_checkpoint_path = os.path.join(args.checkpoint_dir, config.vae_checkpoint)
        if 'Wan2.2_VAE' in config.vae_checkpoint:
            logger.info(f"Loading Wan2.2 VAE: {config.vae_checkpoint}")
            vae = Wan2_2_VAE(vae_pth=vae_checkpoint_path, device=device)
        else:
            logger.info(f"Loading Wan2.1 VAE: {config.vae_checkpoint}")
            vae = Wan2_1_VAE(vae_pth=vae_checkpoint_path, device=device)
        vae.model.eval()
        for p in vae.model.parameters():
            p.requires_grad = False
    else:
        logger.info("Random init mode: skipping T5/VAE loading")

    # ---- Build trainable DiT model ----
    if args.random_init:
        logger.info(f"Constructing WanModel with random init "
                    f"(dim={model_dim}, ffn={model_ffn_dim}, "
                    f"layers={model_num_layers}, heads={model_num_heads})...")
        model = WanModel(
            model_type=config.model_type if hasattr(config, 'model_type') else 't2v',
            patch_size=config.patch_size,
            text_len=config.text_len,
            in_dim=config.in_dim if hasattr(config, 'in_dim') else 16,
            dim=model_dim,
            ffn_dim=model_ffn_dim,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim if hasattr(config, 'text_dim') else 4096,
            out_dim=config.out_dim if hasattr(config, 'out_dim') else 16,
            num_heads=model_num_heads,
            num_layers=model_num_layers,
            window_size=config.window_size,
            qk_norm=config.qk_norm,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
        )
        # Log parameter count
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"WanModel parameter count: {n_params/1e9:.2f}B")
    else:
        logger.info(f"Loading WanModel ({args.model_config})...")
        model = WanModel.from_pretrained(args.checkpoint_dir)
    model = model.to(device=device, dtype=param_dtype)
    model.train()
    model.requires_grad_(True)

    # ---- Replace nn.Linear with TE Linear (if requested) ----
    if args.fuse_wgrad_accumulation and not args.use_te_linear:
        raise ValueError("--fuse-wgrad-accumulation requires --use-te-linear")
    if args.fuse_wgrad_accumulation and args.fsdp_backend == "pytorch":
        raise ValueError(
            "--fuse-wgrad-accumulation is only supported with --fsdp-backend megatron. "
            "PyTorch FSDP2 has no main_grad buffer mechanism for TE to write into."
        )
    if args.fuse_wgrad_accumulation:
        if not args.grad_reduce_in_fp32:
            logger.info("Enabling --grad-reduce-in-fp32 (required by fuse_wgrad_accumulation)")
            args.grad_reduce_in_fp32 = True

    if args.use_te_linear:
        logger.info(f"Replacing nn.Linear → te.Linear "
                    f"(fuse_wgrad_accumulation={args.fuse_wgrad_accumulation})")
        model = replace_linear_with_te(
            model,
            fuse_wgrad_accumulation=args.fuse_wgrad_accumulation,
            alignment=16,
        )

    # ---- FSDP wrapping (backend-dependent) ----
    if args.fsdp_backend == "megatron":
        from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard

        # Megatron-FSDP: optimizer BEFORE fully_shard (re-registers params internally)
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        logger.info("Wrapping with Megatron-FSDP...")
        logger.info(f"Zero DP strategy: {args.zero_dp_strategy}")
        model, optimizer = fully_shard(
            module=model,
            optimizer=optimizer,
            device_mesh=device_mesh,
            dp_shard_dim=DP_SHARD_CP,
            dp_outer_dim=DP_OUTER if use_hsdp else None,
            tp_dim=TP,
            hybrid_fsdp_group=device_mesh[HSDP].get_group() if use_hsdp else None,
            fsdp_unit_modules=[WanAttentionBlock],
            zero_dp_strategy=args.zero_dp_strategy,
            outer_dp_sharding_strategy=args.outer_dp_strategy if use_hsdp else "no_shard",
            preserve_fp32_weights=args.preserve_fp32_weights,
            grad_reduce_in_fp32=args.grad_reduce_in_fp32,
            sync_model_each_microbatch=False,  # Essential for gradient accumulation
            overlap_grad_reduce=args.overlap_grad_reduce,
            overlap_param_gather=args.overlap_param_gather,
            fsdp_double_buffer=args.fsdp_double_buffer,
            nccl_ub=args.nccl_ub,
            disable_symmetric_registration=args.disable_symmetric_registration,
        )
        logger.info("Megatron-FSDP ready")

    else:  # pytorch FSDP2
        from torch.distributed._composable.fsdp import fully_shard as fsdp2_fully_shard
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        reduce_dtype = torch.float32 if args.grad_reduce_in_fp32 else None
        mp_policy = MixedPrecisionPolicy(
            reduce_dtype=reduce_dtype,
        )

        dp_mesh = init_device_mesh("cuda", (world_size,))

        logger.info(f"Wrapping with PyTorch FSDP2 (model dtype={param_dtype}, "
                    f"reduce_dtype={reduce_dtype})...")

        for block in model.blocks:
            fsdp2_fully_shard(block, mesh=dp_mesh, mp_policy=mp_policy)
        
        fsdp2_fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)

        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        logger.info("PyTorch FSDP2 ready")

    # ---- Dataset + infinite iterator ----
    if not args.random_init:
        dataset = MockDataset(args.num_samples, args.frame_num, tuple(args.resolution))
        sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
                   if world_size > 1 else None)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        data_iter = cycle(iter(dataloader))
    else:
        data_iter = None  # random-init generates data directly on GPU

    scheduler = get_cosine_scheduler(optimizer, args.warmup_steps, args.max_steps)
    criterion = FlowMatchingLoss(num_timesteps=config.num_train_timesteps)

    # ---- Train ----
    logger.info("Starting training...")
    train(model, text_encoder, vae, data_iter, optimizer, scheduler, criterion,
          args, rank, device, param_dtype, config)

    logger.info("Training completed!")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
