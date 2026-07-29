import random

from chaoslab.injectors import (
    ContextTruncationInjector,
    OutputTruncationInjector,
    Stage,
    UnicodePerturbationInjector,
)
from chaoslab.registry import build_injector


def rng(seed="t"):
    return random.Random(seed)


def _docs():
    return [{"id": f"d{i}", "text": "word " * 60} for i in range(4)]


def test_context_truncation_drops_and_clips():
    inj = ContextTruncationInjector(intensity=0.5)
    out, ev = inj.apply(_docs(), rng("c"))
    assert ev.stage == Stage.RETRIEVAL.value
    assert len(out) < 4  # some documents dropped
    assert any(d.get("truncated") for d in out)  # survivors clipped


def test_context_truncation_full_intensity_starves():
    inj = ContextTruncationInjector(intensity=1.0)
    out, ev = inj.apply(_docs(), rng("c"))
    assert out == []
    assert ev.meta["starved"] is True


def test_context_truncation_ignores_empty():
    inj = ContextTruncationInjector(intensity=0.9)
    out, ev = inj.apply([], rng("c"))
    assert ev is None and out == []


def test_output_truncation_loses_tail():
    inj = OutputTruncationInjector(intensity=0.7)
    text = "The capital fact appears at the very end: PROCEED."
    out, ev = inj.apply(text, rng("o"))
    assert out.endswith("…")
    assert "PROCEED" not in out
    assert ev.meta["kept_chars"] < len(text)


def test_output_truncation_zero_intensity_is_noop():
    inj = OutputTruncationInjector(intensity=0.0)
    out, ev = inj.apply("hello world", rng("o"))
    assert ev is None and out == "hello world"


def test_unicode_perturbation_swaps_homoglyphs():
    inj = UnicodePerturbationInjector(intensity=1.0)
    out, ev = inj.apply("a chaos cocoa", rng("u"))
    assert out != "a chaos cocoa"
    assert ev.meta["homoglyphs"] > 0
    # visually similar but not byte-identical
    assert len(out) >= len("a chaos cocoa")


def test_unicode_perturbation_ignores_non_string():
    inj = UnicodePerturbationInjector(intensity=1.0)
    out, ev = inj.apply(12345, rng("u"))
    assert ev is None


def test_new_injectors_registered_and_deterministic():
    for name in ("context_truncation", "output_truncation", "unicode_perturbation"):
        a = build_injector(name, intensity=0.6)
        b = build_injector(name, intensity=0.6)
        payload = _docs() if name == "context_truncation" else "deterministic answer text here"
        out_a, _ = a.apply(payload if name != "context_truncation" else _docs(), rng("x"))
        out_b, _ = b.apply(payload if name != "context_truncation" else _docs(), rng("x"))
        assert out_a == out_b
