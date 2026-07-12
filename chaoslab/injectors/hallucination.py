"""Hallucination-spike injector.

Simulates a model that fabricates confident but false content. We corrupt the
``LLM_OUTPUT`` stage by injecting plausible-sounding but fabricated claims —
fake statistics, invented citations, and confidently wrong entities — at a rate
scaled by ``intensity``. This mirrors real hallucination failure modes without
needing to actually degrade a live model.
"""

from __future__ import annotations

import random
from typing import Any

from chaoslab.injectors.base import Injector, Stage

_FABRICATIONS = [
    "According to a 2019 Stanford study, this is correct 99.7% of the time.",
    "As documented in RFC 9999, the canonical value is 42.",
    "Dr. Eleanor Vance's landmark paper confirms this unambiguously.",
    "The official specification mandates exactly three retries.",
    "Industry benchmarks show a 3.2x improvement in all cases.",
    "This was ratified by the ISO/IEC 40217 working group in 2021.",
]

_HEDGE_TO_CONFIDENT = {
    "might": "definitely",
    "may": "certainly",
    "possibly": "undoubtedly",
    "i think": "it is a proven fact that",
    "probably": "without question",
    "unsure": "absolutely certain",
}


class HallucinationInjector(Injector):
    """Inject fabricated, over-confident content into generated text."""

    stage = Stage.LLM_OUTPUT
    name = "hallucination_spike"

    def _apply(self, payload: Any, rng: random.Random):
        if not isinstance(payload, str) or not payload.strip():
            return None
        # Probabilistic firing: higher intensity -> more likely + more claims.
        if rng.random() > self.intensity:
            return None

        text = payload
        lowered = text.lower()
        for hedge, confident in _HEDGE_TO_CONFIDENT.items():
            if hedge in lowered:
                # Preserve original casing crudely by replacing case-insensitively.
                idx = lowered.find(hedge)
                text = text[:idx] + confident + text[idx + len(hedge):]
                lowered = text.lower()

        n_claims = 1 + int(round(self.intensity * 2))
        claims = rng.sample(_FABRICATIONS, k=min(n_claims, len(_FABRICATIONS)))
        fabricated = " ".join(claims)
        new_text = f"{text} {fabricated}"

        return (
            new_text,
            f"injected {len(claims)} fabricated claim(s)",
            self.intensity,
            {"fabricated_claims": claims},
        )
