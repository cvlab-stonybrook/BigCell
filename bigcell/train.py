"""Train the BigCell tile generator.

    torchrun --nproc_per_node 4 -m bigcell.train configs/bigcell/2048.yaml \
        --data-dir /path/to/bigcell_data --exp-name bigcell_2048

Any config key can be overridden on the command line (``--batch-size 8``).
Progressive training (paper Table 6): 256 -> 512 -> 1024 -> 2048, each stage initialised
from the previous one with ``--init-from``.
"""

import argparse
import json
import logging
import math
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from bigcell.data import BigCellTileDataset
from bigcell.distributed import cleanup_distributed, setup_distributed
from bigcell.models import transformer_weights
from bigcell.models.sit import build_sit, flow_matching_loss

# parameters whose shape depends on the resolution; re-initialised when changing resolution
RESOLUTION_DEPENDENT = ("pos_embed", "y_embedder.y_embedding", "y_pos_embed")
# trained alone during the first `freeze_steps` after a resolution change
WARMUP_MODULES = ("y_embedder", "cross_attn")


def get_logger(out_dir, rank):
    logger = logging.getLogger("bigcell")
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
        e.lerp_(p.detach(), 1 - decay)


def load_init_weights(model, path, logger):
    """Initialise from a lower-resolution model (released pipeline folder, .safetensors, or a training .pt)."""
    path = Path(path)
    if path.is_dir():  # a released pipeline folder
        path = transformer_weights(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["ema"]
    own = model.state_dict()
    state = {k: v for k, v in state.items() if not (k.startswith(RESOLUTION_DEPENDENT) and own[k].shape != v.shape)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"init from {path}; re-initialised: {missing}; unexpected: {unexpected}")


def set_trainable(model, warmup_only):
    for name, p in model.named_parameters():
        p.requires_grad = (not warmup_only) or any(m in name for m in WARMUP_MODULES)
    # pos embeddings are fixed sin-cos tables
    model.pos_embed.requires_grad = False
    if model.y_pos_embed is not None:
        model.y_pos_embed.requires_grad = False


def main(args):
    device, rank, world = setup_distributed()
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(args.output_dir) / args.exp_name
    ckpt_dir = out_dir / "checkpoints"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    if world > 1:
        dist.barrier()
    logger = get_logger(out_dir, rank)

    model = build_sit(args.resolution, model=args.model, class_dropout_prob=args.cfg_prob,
                      gradient_checkpointing=args.gradient_checkpointing)
    if args.init_from:
        load_init_weights(model, args.init_from, logger)
    model.to(device)
    ema = deepcopy(model).requires_grad_(False)
    logger.info(f"SiT parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # resume from the latest checkpoint of this experiment
    step, resume = 0, None
    ckpts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    if ckpts:
        resume = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        ema.load_state_dict(resume["ema"])
        step = resume["step"]
        logger.info(f"resumed from {ckpts[-1]} at step {step}")

    # All parameters are given to the optimiser; during the warm-up phase the frozen ones have
    # requires_grad=False, receive no gradient and are therefore not updated by AdamW.
    set_trainable(model, warmup_only=step < args.freeze_steps)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "pos_embed" not in n],
        lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay, eps=args.adam_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, s / max(1, args.warmup_steps)))
    if resume is not None:
        optimizer.load_state_dict(resume["opt"])
        scheduler.load_state_dict(resume["scheduler"])
        del resume

    ddp = DDP(model, device_ids=[device.index])

    dataset = BigCellTileDataset(args.data_dir, split=args.split, resolution=args.resolution)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    local_bs = args.batch_size // world
    loader = DataLoader(dataset, batch_size=local_bs, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    steps_per_epoch = len(loader)
    max_steps = args.max_train_steps or math.ceil(args.epochs * steps_per_epoch)
    logger.info(f"{len(dataset):,} samples, global batch {local_bs * world}, {steps_per_epoch} steps/epoch, "
                f"training for {max_steps} steps")

    if rank == 0 and args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args), resume="allow")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[args.mixed_precision]
    ddp.train()
    epoch = step // steps_per_epoch
    while step < max_steps:
        sampler.set_epoch(epoch)
        for latents, uni in loader:
            if step == args.freeze_steps and args.freeze_steps > 0:
                logger.info("unfreezing all parameters")
                set_trainable(model, warmup_only=False)
                # DDP only syncs gradients of parameters that required grad when it was built
                ddp = DDP(model, device_ids=[device.index])

            latents, uni = latents.to(device, non_blocking=True), uni.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=dtype, enabled=args.mixed_precision != "no"):
                loss = flow_matching_loss(ddp, latents, uni, args.weighting, args.alpha_shift).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema, model, args.ema_decay)
            step += 1

            if step % args.log_every == 0:
                loss_avg = loss.detach().clone()
                if world > 1:
                    dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG)
                logger.info(f"step {step} loss {loss_avg.item():.4f} grad_norm {grad_norm.item():.3f} "
                            f"lr {scheduler.get_last_lr()[0]:.2e}")
                if rank == 0 and args.wandb:
                    wandb.log({"train/loss": loss_avg.item(), "train/lr": scheduler.get_last_lr()[0]}, step=step)

            if rank == 0 and (step % args.checkpoint_every == 0 or step == max_steps):
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "step": step, "args": vars(args)},
                           ckpt_dir / f"{step:07d}.pt")
                logger.info(f"saved {ckpt_dir / f'{step:07d}.pt'}")
            if step >= max_steps:
                break
        epoch += 1

    logger.info("done")
    cleanup_distributed()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?", help="YAML file with default values for the options below")
    # experiment
    p.add_argument("--exp-name", default=None)
    p.add_argument("--output-dir", default="exps")
    p.add_argument("--init-from", default=None, help="lower-resolution model to initialise from")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="bigcell")
    # data
    p.add_argument("--data-dir", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--num-workers", type=int, default=8)
    # model
    p.add_argument("--model", default="SiT-XL/2")
    p.add_argument("--resolution", type=int, default=2048, choices=[256, 512, 1024, 2048])
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--cfg-prob", type=float, default=0.1)
    # flow matching
    p.add_argument("--weighting", default="uniform", choices=["uniform", "shifted_uniform"])
    p.add_argument("--alpha-shift", type=float, default=1.0)
    # optimisation
    p.add_argument("--batch-size", type=int, default=360, help="global batch size")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-train-steps", type=int, default=None, help="overrides --epochs")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-epsilon", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--freeze-steps", type=int, default=0,
                   help="train only the conditioning embedder + cross-attention for the first N steps")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--mixed-precision", default="bf16", choices=["bf16", "fp16", "no"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=5000)

    args, _ = p.parse_known_args()
    if args.config:
        cfg = yaml.safe_load(open(args.config))
        unknown = set(cfg) - {a.dest for a in p._actions}
        if unknown:
            p.error(f"unknown keys in {args.config}: {sorted(unknown)}")
        p.set_defaults(**cfg)
    args = p.parse_args()
    if args.exp_name is None or args.data_dir is None:
        p.error("--exp-name and --data-dir are required (in the config or on the command line)")
    return args


if __name__ == "__main__":
    main(parse_args())
