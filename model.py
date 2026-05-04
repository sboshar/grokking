"""Small decoder-only transformer for grokking.

Defaults follow Nanda et al., "Progress measures for grokking via mechanistic
interpretability" (2023): 1 layer, 4 heads, d_model=128, d_ff=512.

The model is plain pre-norm transformer with learned token + position
embeddings, causal self-attention, and an untied unembedding head. It works
on any device; just call `.to(device)` and the registered buffers will follow.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int = 3
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    d_ff: int = 512
    dropout: float = 0.0


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class GrokkingTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Causal mask: True = "block this attention edge".
        # Registered as a non-persistent buffer so it follows .to(device) but
        # is not saved into the state_dict.
        mask = torch.triu(
            torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) LongTensor of token ids -> logits (B, T, vocab)."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]  # (B, T, d_model)
        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.unembed(x)


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    cfg = ModelConfig(vocab_size=98)
    model = GrokkingTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.seq_len))
    logits = model(x)
    print(f"input  {tuple(x.shape)}")
    print(f"logits {tuple(logits.shape)}")
    print(f"params {num_parameters(model):,}")
