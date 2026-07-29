"""Higher-order resilience analyses built on the experiment engine.

* :mod:`chaoslab.analysis.recovery` — transient chaos + time-to-recover (MTTR).
* :mod:`chaoslab.analysis.search` — adaptive search for the resilience threshold.
"""

from chaoslab.analysis.recovery import RecoveryReport, run_transient
from chaoslab.analysis.search import BreakingPointReport, find_breaking_point

__all__ = [
    "RecoveryReport",
    "run_transient",
    "BreakingPointReport",
    "find_breaking_point",
]
