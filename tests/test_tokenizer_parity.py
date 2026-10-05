"""Tokenizer tests.

The load-bearing invariant: tokenizing segment-by-segment and concatenating equals
tokenizing the rendered string as a whole. If that broke, the loss mask would be
misaligned with the token sequence — silently, and only on some samples.
"""

from __future__ import annotations

from refinery.compiler.render import render_messages, seed_messages, segments
from refinery.trainer.tokenizer import SPECIALS, TOKEN_PATTERN, WordTokenizer

TASK = {
    "hypothesis": "A boy is playing a guitar.",
    "label": "entailed",
    "evidence": [
        {"index": 0, "text": "A boy sits on a stool."},
        {"index": 1, "text": "He strums a guitar, loudly."},
    ],
}


def _trajectory() -> list[dict]:
    msgs = seed_messages(TASK)
    msgs.append({"role": "assistant", "content": "PLAN: read both.\nlookup(0) lookup(1)"})
    msgs.append({"role": "tool", "content": "A boy sits on a stool.", "evidence_index": 0, "tool_ok": True})
    msgs.append({"role": "assistant", "content": "PLAN: compare.\nCITE: 0, 1\nANSWER: entailed"})
    return msgs


def _seg_dicts(msgs):
    return [{"text": s.text, "trainable": s.trainable, "role": s.role, "kind": s.kind}
            for s in segments(msgs)]


def test_segment_tokenization_equals_whole_text_tokenization():
    tok = WordTokenizer.build([render_messages(_trajectory())], max_vocab=8000)
    msgs = _trajectory()
    piecewise, _mask = tok.encode_segments(_seg_dicts(msgs))
    whole = tok.encode(render_messages(msgs))
    assert piecewise == whole


def test_specials_are_never_oov():
    tok = WordTokenizer.build([render_messages(_trajectory())], max_vocab=8000)
    for special in SPECIALS:
        assert special in tok.stoi


def test_specials_survive_a_tiny_vocab_budget():
    """Reserving special slots matters: with max_vocab <= len(SPECIALS) the model
    would have no ids for the turn delimiters at all."""
    tok = WordTokenizer.build([render_messages(_trajectory())], max_vocab=3)
    assert set(SPECIALS) <= set(tok.stoi)


def test_vocab_is_a_pure_function_of_the_text():
    text = "a b c a b"
    assert WordTokenizer.build([text]).itos == WordTokenizer.build([text]).itos


def test_oov_tokens_are_dropped_not_mapped_to_unk():
    tok = WordTokenizer.build(["hello world"], max_vocab=8000)
    ids = tok.encode("hello zzzzqqq world")
    assert tok.encode("hello world") == ids


def test_loss_mask_marks_only_trainable_segments():
    tok = WordTokenizer.build([render_messages(_trajectory())], max_vocab=8000)
    segs = _seg_dicts(_trajectory())
    ids, mask = tok.encode_segments(segs)

    trainable_text = "".join(s["text"] for s in segs if s["trainable"])
    masked_text = " ".join(tok.itos[i] for i, m in zip(ids, mask, strict=True) if m)
    assert len(masked_text) > 0
    assert "lookup" in masked_text
    assert "<|im_start|>" not in masked_text
    assert trainable_text  # sanity


def test_kinds_allow_narrows_the_mask_for_curriculum_stage_1():
    tok = WordTokenizer.build([render_messages(_trajectory())], max_vocab=8000)
    segs = _seg_dicts(_trajectory())
    for seg in segs:
        seg["kinds_allow"] = ["tool_call"]
    _ids, mask = tok.encode_segments(segs)
    supervised = [i for i, m in zip(tok.encode(render_messages(_trajectory())), mask, strict=True) if m]
    assert supervised, "stage 1 must still supervise something"
    # Only tool-call tokens: no ANSWER/CITE tokens survive.
    decoded = " ".join(tok.itos[i] for i in supervised)
    assert "lookup" in decoded
    assert "ANSWER" not in decoded


def test_token_pattern_keeps_newlines_and_punctuation():
    assert TOKEN_PATTERN.findall("a, b.\nc") == ["a", ",", "b", ".", "\n", "c"]
