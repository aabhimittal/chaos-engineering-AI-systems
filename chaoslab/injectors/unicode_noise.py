"""Unicode / homoglyph perturbation injector.

Models an under-tested industrial robustness gap: input text that *looks*
identical to a human but is a different byte sequence. Homoglyph confusables
(Latin ``a`` → Cyrillic ``а``) and invisible zero-width characters routinely
appear in user input — copy-pasted from the web, injected by an adversary to
evade keyword filters, or produced by a broken encoding pipeline.

Injected at the ``PROMPT`` stage, it corrupts the query's tokens so that exact
retrieval/matching degrades even though the text is visually unchanged — the
kind of silent failure that unicode-naive systems ship to production.
"""

from __future__ import annotations

import random
from typing import Any

from chaoslab.injectors.base import Injector, Stage

# Latin -> visually-identical confusable (mostly Cyrillic / Greek).
_HOMOGLYPHS = {
    "a": "а", "c": "с", "e": "е", "i": "і", "j": "ј",
    "o": "о", "p": "р", "s": "ѕ", "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т",
}
_ZERO_WIDTH = "​"  # zero-width space


class UnicodePerturbationInjector(Injector):
    """Swap characters for homoglyphs and sprinkle zero-width spaces."""

    stage = Stage.PROMPT
    name = "unicode_perturbation"

    def _apply(self, payload: Any, rng: random.Random):
        if not isinstance(payload, str) or not payload:
            return None

        swapped = 0
        invisibles = 0
        out = []
        for ch in payload:
            if ch in _HOMOGLYPHS and rng.random() < self.intensity:
                out.append(_HOMOGLYPHS[ch])
                swapped += 1
            else:
                out.append(ch)
            # Occasionally inject an invisible character between glyphs.
            if rng.random() < 0.15 * self.intensity:
                out.append(_ZERO_WIDTH)
                invisibles += 1

        if swapped == 0 and invisibles == 0:
            return None
        return (
            "".join(out),
            f"swapped {swapped} homoglyph(s), inserted {invisibles} zero-width char(s)",
            self.intensity,
            {"homoglyphs": swapped, "zero_width": invisibles},
        )
