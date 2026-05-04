"""Training loop for grokking experiments.

Reproduces the canonical grokking curve: train accuracy snaps to ~100%
quickly, val accuracy stays near chance for a long time, then suddenly
jumps to ~100% well after the train loss has plateaued.

Defaults (p=97, full-batch AdamW, weight_decay=1.0, lr=1e-3) match the
setup used by Nanda et al. (2023). On a single A100/H100, full grok on
addition takes a few minutes of wallclock per ~10k steps.

Usage:
    python train.py                              # defaults
    python train.py --op mul --n_steps 100000    # multiplication, longer
    python train.py --device cuda --out_dir runs/exp1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import GrokkingTask
from model import GrokkingTransformer, ModelConfig, num_parameters


@dataclass
class TrainConfig:
    # Task
    p: int = 97
    op: str = "add"
    train_frac: float = 0.3

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    d_ff: int = 512

    # Optim
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98

    # Loop
    n_steps: int = 50_000
    batch_size: int = 1_000_000  # >= dataset -> full-batch
    log_every: int = 100
    save_every: int = 0  # 0 = only save at end

    # Misc
    seed: int = 0
    out_dir: str = "runs/default"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:0" | "mps"


def get_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor) -> tuple[float, float]:
    """Full-batch eval at the final ('=') position."""
    was_training = model.training
    model.eval()
    logits = model(X)[:, -1, :]
    loss = F.cross_entropy(logits, Y).item()
    acc = (logits.argmax(-1) == Y).float().mean().item()
    if was_training:
        model.train()
    return loss, acc


def train(cfg: TrainConfig) -> None:
    device = get_device(cfg.device)
    print(f"device: {device}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # ---- Data ----
    task = GrokkingTask(p=cfg.p, op=cfg.op, train_frac=cfg.train_frac, seed=cfg.seed)
    train_ds, val_ds = task.split()

    # Move to device once; everything fits comfortably for p ~ 100.
    train_X = train_ds.tensors[0].to(device)
    train_Y = train_ds.tensors[1].to(device)
    val_X = val_ds.tensors[0].to(device)
    val_Y = val_ds.tensors[1].to(device)
    print(f"task: {cfg.op} mod {cfg.p}  train={len(train_ds)}  val={len(val_ds)}")

    full_batch = cfg.batch_size >= len(train_ds)
    if full_batch:
        print(f"full-batch training (batch_size={len(train_ds)})")
        loader_iter = None
    else:
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        loader_iter = iter(loader)
        print(f"minibatch training (batch_size={cfg.batch_size})")

    # ---- Model ----
    model_cfg = ModelConfig(
        vocab_size=task.vocab_size,
        seq_len=task.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
    )
    model = GrokkingTransformer(model_cfg).to(device)
    print(f"model: {num_parameters(model):,} params")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

    # ---- Loop ----
    metrics: list[dict] = []
    t0 = time.time()

    for step in range(cfg.n_steps + 1):
        if full_batch:
            X, Y = train_X, train_Y
        else:
            try:
                X, Y = next(loader_iter)  # type: ignore[arg-type]
            except StopIteration:
                loader_iter = iter(loader)
                X, Y = next(loader_iter)
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)

        logits = model(X)[:, -1, :]
        loss = F.cross_entropy(logits, Y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % cfg.log_every == 0:
            train_loss, train_acc = evaluate(model, train_X, train_Y)
            val_loss, val_acc = evaluate(model, val_X, val_Y)
            elapsed = time.time() - t0
            row = {
                "step": step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "elapsed_s": elapsed,
            }
            metrics.append(row)
            print(
                f"step {step:6d}  "
                f"train loss {train_loss:.4f} acc {train_acc:.3f}  |  "
                f"val loss {val_loss:.4f} acc {val_acc:.3f}  |  "
                f"{elapsed:6.0f}s"
            )
            # Persist incrementally so a kill -9 doesn't lose history.
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            torch.save(model.state_dict(), out_dir / f"model_step{step}.pt")

    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"done. artifacts -> {out_dir}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for field, default in asdict(defaults).items():
        t = type(default)
        if t is bool:
            parser.add_argument(
                f"--{field}",
                type=lambda x: str(x).lower() in {"1", "true", "yes"},
                default=default,
            )
        else:
            parser.add_argument(f"--{field}", type=t, default=default)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
