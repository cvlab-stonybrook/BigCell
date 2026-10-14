"""Train the conditioning generation model (CGM).

    torchrun --nproc_per_node 4 -m cgm.train configs/cgm/8192.yaml \
        --data-dir /path/to/bigcell_data --titan-text-dir /path/to/titan_text

The released model was trained at 4096 (16x16 grids) and then fine-tuned at 8192 (32x32 grids)
with ``--init-from`` (position-dependent buffers are re-initialised).
"""

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from bigcell.distributed import cleanup_distributed, setup_distributed
from cgm.data import GridTextDataset
from cgm.models import CGM_B, GridNormalizer, default_time_shift, flow_matching_loss


def get_logger(out_dir, rank):
    logger = logging.getLogger("cgm")
    if rank == 0 and not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        for h in (logging.StreamHandler(), logging.FileHandler(out_dir / "log.txt")):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        if p.requires_grad:
            e.lerp_(p.detach(), 1 - decay)


def lr_lambda(warmup_steps, decay_end_steps, final_ratio, warmup_from_zero):
    """Warm-up (constant, or linear from zero), then linear decay to final_ratio."""
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps if warmup_from_zero else 1.0
        if step >= decay_end_steps:
            return final_ratio
        progress = (step - warmup_steps) / max(1, decay_end_steps - warmup_steps)
        return 1.0 - (1.0 - final_ratio) * progress
    return fn


def load_init_weights(model, path, logger):
    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)["ema"]
    own = model.state_dict()
    state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"init from {path}; re-initialised (shape mismatch): {missing}")


def main(args):
    device, rank, world = setup_distributed()
    torch.manual_seed(args.seed * world + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.output_dir) / args.exp_name
    ckpt_dir = out_dir / "checkpoints"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    if world > 1:
        dist.barrier()
    logger = get_logger(out_dir, rank)

    grid_size = args.size // 256
    model = CGM_B(input_size=grid_size, class_dropout_prob=args.cfg_prob)
    if args.init_from:
        load_init_weights(model, args.init_from, logger)
    model.to(device)
    ema = deepcopy(model).requires_grad_(False)
    normalizer = GridNormalizer(Path(args.data_dir) / "image_embedding_stats" / f"uni2_{args.size}" /
                                "normalization_stats.pt").to(device)
    shift = default_time_shift(grid_size)
    logger.info(f"CGM parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M, time shift {shift:.2f}")

    dataset = GridTextDataset(args.data_dir, args.titan_text_dir, split=args.split, size=args.size,
                              manifest=args.text_manifest)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    steps_per_epoch = len(loader) // args.grad_accum_steps
    logger.info(f"{len(dataset):,} samples; global batch {args.batch_size * world * args.grad_accum_steps}; "
                f"{steps_per_epoch} optimiser steps/epoch")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(
        args.warmup_epochs * steps_per_epoch, args.decay_end_epoch * steps_per_epoch,
        args.final_lr / args.learning_rate, args.warmup_from_zero))

    start_epoch, step = 0, 0
    ckpts = sorted(ckpt_dir.glob("ep-*.pt")) if ckpt_dir.exists() else []
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, step = ckpt["epoch"], ckpt["step"]
        logger.info(f"resumed from {ckpts[-1]} (epoch {start_epoch}, step {step})")
        del ckpt

    ddp = DDP(model, device_ids=[device.index])
    if rank == 0 and args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args), resume="allow")

    def save(epoch):
        if rank == 0:
            path = ckpt_dir / f"ep-{epoch:07d}.pt"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step, "args": vars(args)}, path)
            logger.info(f"saved {path}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[args.mixed_precision]
    ddp.train()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        running, n_micro = 0.0, 0
        for i, (grid, text) in enumerate(loader):
            grid, text = grid.to(device, non_blocking=True), text.to(device, non_blocking=True)
            with torch.no_grad():
                x0 = normalizer.encode(grid)
            last_micro = (i + 1) % args.grad_accum_steps == 0
            # only all-reduce gradients on the last micro-batch of an accumulation cycle
            with ddp.no_sync() if not last_micro else torch.enable_grad():
                with torch.autocast("cuda", dtype=dtype, enabled=args.mixed_precision != "no"):
                    loss = flow_matching_loss(ddp, x0, text, shift).mean()
                (loss / args.grad_accum_steps).backward()
            running += loss.item()
            n_micro += 1
            if not last_micro:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema, model, args.ema_decay)
            step += 1

            if step % args.log_every == 0:
                avg = torch.tensor(running / n_micro, device=device)
                if world > 1:
                    dist.all_reduce(avg, op=dist.ReduceOp.AVG)
                logger.info(f"epoch {epoch} step {step} loss {avg.item():.4f} lr {scheduler.get_last_lr()[0]:.2e}")
                if rank == 0 and args.wandb:
                    wandb.log({"train/loss": avg.item(), "train/lr": scheduler.get_last_lr()[0]}, step=step)
                running, n_micro = 0.0, 0
            if args.max_train_steps and step >= args.max_train_steps:
                break
        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch + 1 == args.epochs or (
                args.max_train_steps and step >= args.max_train_steps):
            save(epoch + 1)
        if args.max_train_steps and step >= args.max_train_steps:
            break

    logger.info("done")
    cleanup_distributed()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?")
    p.add_argument("--exp-name", default=None)
    p.add_argument("--output-dir", default="exps")
    p.add_argument("--init-from", default=None, help="CGM trained at a smaller region size to fine-tune from")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="bigcell-cgm")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--titan-text-dir", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--text-manifest", default="image_list_text_titan.h5", help="file in <data-dir>/metadata")
    p.add_argument("--size", type=int, default=8192, choices=[4096, 8192, 16384])
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--cfg-prob", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8, help="per-GPU micro-batch")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1400)
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--final-lr", type=float, default=2e-5)
    p.add_argument("--warmup-epochs", type=float, default=40)
    p.add_argument("--warmup-from-zero", action="store_true")
    p.add_argument("--decay-end-epoch", type=float, default=800)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--mixed-precision", default="no", choices=["bf16", "fp16", "no"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--checkpoint-every-epochs", type=int, default=5)

    args, _ = p.parse_known_args()
    if args.config:
        cfg = yaml.safe_load(open(args.config))
        unknown = set(cfg) - {a.dest for a in p._actions}
        if unknown:
            p.error(f"unknown keys in {args.config}: {sorted(unknown)}")
        p.set_defaults(**cfg)
    args = p.parse_args()
    if not (args.exp_name and args.data_dir and args.titan_text_dir):
        p.error("--exp-name, --data-dir and --titan-text-dir are required")
    return args


if __name__ == "__main__":
    main(parse_args())
