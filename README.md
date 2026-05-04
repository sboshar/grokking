# Grokking

Reproducing the modular-arithmetic results from Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets* (2022).

## Goal

Train a small transformer on tasks of the form

```
a <op> b mod p   →   c
```

and observe the canonical grokking phenomenon: train accuracy hits ~100% quickly, validation accuracy stays near chance for a long time, then suddenly snaps to ~100% well after the train loss has converged.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Plan

- `data.py` — generate `(a, b, a∘b mod p)` triples for a chosen binary op, split into train / val.
- `model.py` — small decoder-only transformer (a few layers, single head is enough for `p ≈ 97`).
- `train.py` — full-batch (or large-batch) AdamW with weight decay; log train/val accuracy each step; run long enough to see the val curve snap.
- `notebooks/` — analysis (loss curves, embedding visualizations).

## References

- Power, Burda, Edwards, Babuschkin, Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* 2022. https://arxiv.org/abs/2201.02177
- Nanda et al. *Progress measures for grokking via mechanistic interpretability.* 2023. https://arxiv.org/abs/2301.05217
