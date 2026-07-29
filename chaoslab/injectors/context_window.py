"""Context-window / token-budget failure injectors.

Real LLM systems fail at their *boundaries*, not just their logic. Two very
common, under-tested industrial failure modes:

* :class:`ContextTruncationInjector` — the retrieved context exceeds the model's
  context window (or an aggressive truncation policy kicks in), so documents are
  dropped and the survivors are clipped. The answer-bearing passage can vanish
  entirely, silently degrading a RAG system.
* :class:`OutputTruncationInjector` — generation hits ``max_tokens`` or a
  streaming connection is cut, so the answer is truncated mid-thought. Fluent,
  plausible, and missing the payload — a failure users rarely notice in eval.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from chaoslab.injectors.base import Injector, Stage


class ContextTruncationInjector(Injector):
    """Drop trailing retrieved documents and clip survivors (context overflow)."""

    stage = Stage.RETRIEVAL
    name = "context_truncation"

    def _apply(self, payload: Any, rng: random.Random):
        if not isinstance(payload, list) or not payload:
            return None

        n = len(payload)
        # Higher intensity keeps fewer documents (can reach zero = starvation).
        keep = int(round(n * (1.0 - self.intensity)))
        keep = max(0, min(n, keep))
        survivors: List[Dict[str, Any]] = []
        # Per-document character budget shrinks with intensity.
        budget = max(1, int(200 * (1.0 - 0.75 * self.intensity)))
        for doc in payload[:keep]:
            d = dict(doc) if isinstance(doc, dict) else {"text": str(doc)}
            text = d.get("text", "")
            if len(text) > budget:
                d["text"] = text[:budget]
                d["truncated"] = True
            survivors.append(d)

        return (
            survivors,
            f"kept {keep}/{n} docs, {budget}-char budget",
            self.intensity,
            {"dropped": n - keep, "char_budget": budget, "starved": keep == 0},
        )


class OutputTruncationInjector(Injector):
    """Truncate generated text, modelling a max_tokens / streaming cutoff."""

    stage = Stage.LLM_OUTPUT
    name = "output_truncation"

    def _apply(self, payload: Any, rng: random.Random):
        if not isinstance(payload, str) or not payload:
            return None
        # Keep a leading fraction of the answer; the tail (often the payload) is lost.
        keep_frac = max(0.05, 1.0 - self.intensity)
        cut = max(1, int(len(payload) * keep_frac))
        if cut >= len(payload):
            return None
        truncated = payload[:cut].rstrip() + "…"
        return (
            truncated,
            f"truncated output to {cut}/{len(payload)} chars",
            self.intensity,
            {"kept_chars": cut, "original_chars": len(payload)},
        )
