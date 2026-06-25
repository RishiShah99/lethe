"""Box driver: DDP PTB-XL training of the ~1.1B Mamba-1 SISO classifier (Phase F.3).

Runs on the 8x B200 under torchrun; consumes the Phase C kernels via
``Mamba3ECGClassifier`` (CUDA path) and the real ``PTBXL`` dataset. Spot-resilient
via ``--resume`` (MedicalTrainer's atomic trainer_state.pt). Logs throughput and a
rough MFU per log interval.

Launch (box):
    export PATH=$HOME/.local/bin:$PATH
    torchrun --standalone --nproc_per_node=8 scratch/ptbxl_train.py \
        --data-root ~/data/ptbxl --steps 20000 --batch-size 8 --resume

Single-process smoke (no torchrun) also works: omits DDP.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler

from flash_mamba_rl.medical.data import PTBXL
from flash_mamba_rl.medical.model import Mamba3Config, Mamba3ECGClassifier
from flash_mamba_rl.medical.train import MedicalTrainConfig, MedicalTrainer

# B200 bf16 dense peak (FLOP/s), per GPU — MFU denominator. Approximate.
_PEAK_FLOPS = 2.25e15


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    # #19 regularization knobs.
    p.add_argument("--config", default="b1", choices=("b1", "b_mid"))
    p.add_argument("--dropout", type=float, default=None, help="override config dropout")
    p.add_argument("--lr-decay", action="store_true", help="cosine LR decay after warmup")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--augment", action="store_true", help="train-only ECG augmentation")
    p.add_argument("--sampling-rate", type=int, default=100, choices=(100, 500))
    p.add_argument("--label-set", default="superclass", choices=("superclass", "subclass"))
    p.add_argument("--checkpoint-dir", default="ptbxl_out")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _init_dist() -> tuple[bool, int, int, torch.device]:
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return True, rank, world, torch.device(f"cuda:{local_rank}")


def _loader(
    ds: PTBXL, batch_size: int, *, dist: bool, world: int, rank: int, workers: int, shuffle: bool
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    sampler = (
        DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle) if dist else None
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=workers,
        drop_last=shuffle,
        pin_memory=True,
    )


def main() -> None:
    args = _parse()
    dist, rank, world, device = _init_dist()

    train_ds = PTBXL(
        args.data_root,
        sampling_rate=args.sampling_rate,
        split="train",
        label_set=args.label_set,
        augment=args.augment,
    )
    val_ds = PTBXL(
        args.data_root,
        sampling_rate=args.sampling_rate,
        split="val",
        label_set=args.label_set,
    )

    base = Mamba3Config.b_mid() if args.config == "b_mid" else Mamba3Config.b1()
    # The scan requires T % chunk_size == 0 (model docstring); T = sampling_rate*10
    # (1000 @ 100 Hz, 5000 @ 500 Hz). The default chunk_size=64 divides neither.
    # Pick the largest divisor <= 64 for good tiling (50 divides both 1000 and 5000).
    seq_t = args.sampling_rate * 10
    chunk_size = next(k for k in range(64, 0, -1) if seq_t % k == 0)
    cfg = Mamba3Config(
        d_model=base.d_model,
        n_layers=base.n_layers,
        d_state=base.d_state,
        conv_kernel_size=base.conv_kernel_size,
        chunk_size=chunk_size,
        eps=base.eps,
        n_classes=train_ds.n_classes,
        n_leads=12,
        dropout=args.dropout if args.dropout is not None else base.dropout,
    )
    # fp32: bf16 SSM training NaNs immediately here (the recurrence + RMSNorm are
    # bf16-fragile) and the 1.1B model leaves ample B200 memory in fp32. bf16/MFU
    # is a later optimisation, separate from the accuracy result.
    model = Mamba3ECGClassifier(cfg).to(torch.float32)

    tcfg = MedicalTrainConfig(
        total_steps=args.steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_decay=args.lr_decay,
        min_lr_ratio=args.min_lr_ratio,
        device=str(device),
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
    )
    trainer = MedicalTrainer(model, tcfg)
    if args.resume and trainer.load_checkpoint() and rank == 0:
        print(f"resumed at step {trainer.step_idx}", flush=True)

    train_loader = _loader(
        train_ds,
        args.batch_size,
        dist=dist,
        world=world,
        rank=rank,
        workers=args.num_workers,
        shuffle=True,
    )
    val_loader = _loader(
        val_ds,
        args.batch_size,
        dist=dist,
        world=world,
        rank=rank,
        workers=args.num_workers,
        shuffle=False,
    )

    # The trainer owns a generic run(); the box wants throughput/MFU per interval,
    # so the loop is inlined here. tokens/step = global_batch * samples-per-record
    # (T = sampling_rate * 10 s); MFU uses the 6N forward+backward heuristic.
    n_params = cfg.analytic_param_count()
    tokens_per_step = args.batch_size * world * (args.sampling_rate * 10)
    data = _cycle(train_loader)
    last = time.perf_counter()
    while trainer.step_idx < tcfg.total_steps:
        ecg, labels = next(data)
        loss, grad_norm = trainer.train_step(ecg, labels)
        if rank == 0 and trainer.step_idx % tcfg.log_every == 0:
            now = time.perf_counter()
            dt = (now - last) / tcfg.log_every
            last = now
            sps = (args.batch_size * world) / dt if dt > 0 else 0.0
            mfu = (6 * n_params * tokens_per_step) / (dt * _PEAK_FLOPS * world) if dt > 0 else 0.0
            print(
                f"step={trainer.step_idx} loss={loss:.4f} grad_norm={grad_norm:.3f} "
                f"samples/s={sps:.1f} mfu={mfu:.3f}",
                flush=True,
            )
        if trainer.step_idx % tcfg.eval_every == 0:
            metrics = trainer.evaluate(val_loader)
            if rank == 0:
                print(f"[val] step={trainer.step_idx} {metrics}", flush=True)
        if trainer.step_idx % tcfg.save_every == 0:
            trainer.save_checkpoint()

    trainer.save_checkpoint()
    if dist:
        torch.distributed.destroy_process_group()


def _cycle(loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]):  # type: ignore[no-untyped-def]
    while True:
        yield from loader


if __name__ == "__main__":
    main()
