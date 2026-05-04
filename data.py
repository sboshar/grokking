"""Modular arithmetic datasets for grokking experiments.

Generates every (a, b, c) triple with c = a OP b (mod p), tokenizes them as
[a, b, =] -> answer, then deterministically splits into train/val by a fixed
seed.

Vocab layout (size = p + 1):
    tokens 0..p-1  -> the numbers themselves
    token  p       -> "=" sentinel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import TensorDataset

Operation = Callable[[int, int, int], int]

OPS: dict[str, Operation] = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
    # `div`: a * b^{-1} mod p; defined only for b != 0 (handled in build_pairs)
    "div": lambda a, b, p: (a * pow(b, -1, p)) % p,
}


@dataclass
class GrokkingTask:
    p: int = 97
    op: str = "add"
    train_frac: float = 0.3
    seed: int = 0

    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @property
    def eq_token(self) -> int:
        return self.p

    @property
    def seq_len(self) -> int:
        # input is [a, b, =] (length 3); model predicts the answer at the "=" position
        return 3

    def build_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize every legal (a, b) -> c triple as token tensors.

        Returns:
            X: LongTensor of shape (N, seq_len), columns = [a, b, =]
            Y: LongTensor of shape (N,), the answer token (in 0..p-1)
        """
        if self.op not in OPS:
            raise ValueError(f"Unknown op {self.op!r}; choose from {list(OPS)}")

        op_fn = OPS[self.op]
        b_start = 1 if self.op == "div" else 0

        a_vals: list[int] = []
        b_vals: list[int] = []
        c_vals: list[int] = []
        for a in range(self.p):
            for b in range(b_start, self.p):
                a_vals.append(a)
                b_vals.append(b)
                c_vals.append(op_fn(a, b, self.p))

        a_t = torch.tensor(a_vals, dtype=torch.long)
        b_t = torch.tensor(b_vals, dtype=torch.long)
        c_t = torch.tensor(c_vals, dtype=torch.long)
        eq = torch.full_like(a_t, self.eq_token)

        X = torch.stack([a_t, b_t, eq], dim=1)  # (N, 3)
        return X, c_t

    def split(self) -> tuple[TensorDataset, TensorDataset]:
        """Random 70/30 split (or whatever train_frac says), seeded."""
        X, Y = self.build_pairs()
        n = X.shape[0]
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g)
        n_train = int(round(self.train_frac * n))
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        return (
            TensorDataset(X[train_idx], Y[train_idx]),
            TensorDataset(X[val_idx], Y[val_idx]),
        )


if __name__ == "__main__":
    task = GrokkingTask(p=97, op="add", train_frac=0.3)
    train, val = task.split()
    print(f"vocab_size = {task.vocab_size}, seq_len = {task.seq_len}")
    print(f"train = {len(train)}, val = {len(val)} (total = {len(train) + len(val)})")
    x, y = train[0]
    print(f"sample: tokens={x.tolist()}  answer={y.item()}")
