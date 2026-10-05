"""Word-level tokenizer for the toy student.

Why word-level and not BPE (LLD.md D-015): at 0.9M non-embedding parameters the
tokenizer must not become the confound. With a word-level vocab the whole
`lookup(3)` call is four tokens, so tool-call syntax is learnable in curriculum
stage 1 at this data scale. A BPE vocab trained on a few hundred trajectories
would be mostly noise, and "the model can't emit tool calls" would become
untestable — indistinguishable from a tokenizer problem.

The vocab is built from the **train split only**. Including eval text would leak
the eval distribution into the student's embedding table, which is a quieter and
easier-to-miss version of trap #1.

Invariant relied on elsewhere: tokenizing segment-by-segment and concatenating
equals tokenizing the rendered string as a whole, because the pattern is local
and segments break only at natural boundaries. `tests/test_tokenizer_parity.py`
asserts it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["WordTokenizer", "SPECIALS", "TOKEN_PATTERN"]

SPECIALS = ("<|pad|>", "<|im_start|>", "<|im_end|>")

#: Ordered alternation: specials first so they are never split, then newline, then
#: words (allowing internal apostrophes), then numbers, then single punctuation
#: characters. Every token is produced independently of its neighbours.
TOKEN_PATTERN = re.compile(
    r"<\|[^|<>]*\|>|\n|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]"
)


@dataclass
class WordTokenizer:
    max_vocab: int = 8000
    itos: list[str] = field(default_factory=list)
    stoi: dict[str, int] = field(default_factory=dict)

    # ---- construction -------------------------------------------------------
    @classmethod
    def build(cls, texts: list[str], *, max_vocab: int = 8000, min_count: int = 1) -> "WordTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(TOKEN_PATTERN.findall(text))

        # Frequency order, ties broken lexicographically so the vocab is a pure
        # function of the input text (reproducibility, not cosmetics).
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = [tok for tok, count in ordered if count >= min_count]

        # Reserve slots for specials so a later special is never OOV.
        budget = max_vocab - len(SPECIALS)
        itos = list(SPECIALS) + kept[:budget]
        return cls(max_vocab=max_vocab, itos=itos, stoi={tok: i for i, tok in enumerate(itos)})

    # ---- properties ---------------------------------------------------------
    @property
    def pad_id(self) -> int:
        return self.stoi["<|pad|>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def is_empty(self) -> bool:
        return not self.itos

    # ---- encoding -----------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """OOV tokens are dropped, not replaced with UNK.

        Dropping is deliberate: an UNK in the middle of a tool call teaches the
        model that tool calls contain UNK. Dropped words simply become invisible
        context, and the loss never lands on a token the model cannot predict.
        """
        return [self.stoi[t] for t in TOKEN_PATTERN.findall(text) if t in self.stoi]

    def encode_segments(self, segments: list[dict]) -> tuple[list[int], list[int]]:
        """Encode segments, returning (ids, loss_mask) with mask aligned to ids.

        `trainable` from the renderer is intersected with the caller's policy, so
        curriculum stage 1 can narrow supervision to `kind == "tool_call"` without
        the tokenizer knowing anything about curricula.
        """
        ids: list[int] = []
        mask: list[int] = []
        for seg in segments:
            policy = seg.get("trainable", False)
            if seg.get("kinds_allow") is not None:
                policy = policy and seg["kind"] in seg["kinds_allow"]
            for tok in TOKEN_PATTERN.findall(seg["text"]):
                tid = self.stoi.get(tok)
                if tid is None:
                    continue
                ids.append(tid)
                mask.append(1 if policy else 0)
        return ids, mask

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))

    # ---- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"max_vocab": self.max_vocab, "itos": self.itos}, ensure_ascii=False),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(max_vocab=data["max_vocab"], itos=data["itos"])
        tok.stoi = {t: i for i, t in enumerate(tok.itos)}
        return tok
