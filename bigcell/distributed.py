import os

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialise NCCL if launched with torchrun; otherwise run on a single GPU.
    Returns (device, rank, world_size)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
        return torch.device("cuda", local_rank), dist.get_rank(), dist.get_world_size()
    return torch.device("cuda"), 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
