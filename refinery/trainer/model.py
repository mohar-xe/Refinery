"""The toy student: a ~0.9M non-embedding-parameter causal transformer.

Architecture choices are in LLD.md D-015/D-015a. The number that gets reported is
`non_embedding_params()` — the embedding table is ~1.3M rows at an 8k vocab, so
quoting the total (≈2.2M) would flatter the model by 2.4x.

Why a hand-rolled stack instead of a pretrained small LM: the point of the toy
run is the pipeline, and a 135M model trained on a few hundred trajectories would
be dominated by its pretraining. A from-scratch model makes every number in the
write-up attributable to the verified trajectories.
"""

from __future__ import annotations

import torch
from torch import nn

from refinery.config import ArchCfg

__all__ = ["TinyTransformer", "build_model"]


class TinyTransformer(nn.Module):
    """Decoder-only transformer trained from scratch.

    Uses `nn.TransformerEncoder` with a causal mask: at this scale that is
    indistinguishable from a hand-written decoder block and far easier to audit,
    which matters more here than the last few percent of throughput.
    """

    def __init__(self, arch: ArchCfg, vocab_size: int) -> None:
        super().__init__()
        if arch.d_model % arch.n_head:
            raise ValueError(f"d_model={arch.d_model} not divisible by n_head={arch.n_head}")

        self.arch = arch
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, arch.d_model)
        self.pos_emb = nn.Embedding(arch.context, arch.d_model)
        self.drop = nn.Dropout(arch.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=arch.d_model,
            nhead=arch.n_head,
            dim_feedforward=arch.d_ff,
            dropout=arch.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=arch.n_layer)
        self.norm_f = nn.LayerNorm(arch.d_model)

        # Tied weights: at 1M params the embedding matrix is a large fraction of
        # the budget, so a separate output head would cost more than it buys.
        self.lm_head = nn.Linear(arch.d_model, vocab_size, bias=False)
        if arch.tied_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def non_embedding_params(self) -> int:
        """Everything except the token embedding table (which is tied to the head)."""
        total = sum(p.numel() for p in self.parameters())
        return total - self.token_emb.weight.numel()

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) int64 -> logits (B, T, V)."""
        b, t = idx.shape
        if t > self.arch.context:
            raise ValueError(f"sequence length {t} exceeds context {self.arch.context}")

        positions = torch.arange(t, device=idx.device)
        hidden = self.drop(self.token_emb(idx) + self.pos_emb(positions).unsqueeze(0))

        # Causal mask: strictly upper-triangular, so position i cannot see > i.
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=idx.device), diagonal=1)
        hidden = self.encoder(hidden, mask=causal)
        return self.lm_head(self.norm_f(hidden))


def build_model(arch: ArchCfg, vocab_size: int) -> TinyTransformer:
    return TinyTransformer(arch, vocab_size)
