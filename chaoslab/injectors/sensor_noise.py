"""Robotics sensor-noise injector.

Simulates degraded perception for embodied/robotics AI by corrupting raw
``SENSOR`` readings. Sensor payloads are dicts of channel -> reading, where a
reading is a scalar or a list of scalars (e.g. lidar ranges, IMU, joint angles).

Failure mechanisms, blended by ``intensity``:

* **Gaussian noise** — additive measurement noise.
* **Dropout** — channels/values randomly go to zero (dead pixels / lost packets).
* **Stuck-at / bias** — a constant offset, modelling a miscalibrated sensor.
"""

from __future__ import annotations

import random
from typing import Any, Dict

from chaoslab.injectors.base import Injector, Stage


class SensorNoiseInjector(Injector):
    """Corrupt robotics sensor readings with noise, dropout, and bias."""

    stage = Stage.SENSOR
    name = "sensor_noise"

    def _noisy(self, value: float, rng: random.Random) -> float:
        sigma = 0.2 * self.intensity * (abs(value) + 1.0)
        if rng.random() < 0.15 * self.intensity:  # dropout
            return 0.0
        bias = 0.1 * self.intensity
        return value + rng.gauss(0, sigma) + bias

    def _apply(self, payload: Any, rng: random.Random):
        if not isinstance(payload, dict) or not payload:
            return None

        corrupted: Dict[str, Any] = {}
        affected = 0
        for channel, reading in payload.items():
            if isinstance(reading, (int, float)):
                corrupted[channel] = self._noisy(float(reading), rng)
                affected += 1
            elif isinstance(reading, list) and all(
                isinstance(x, (int, float)) for x in reading
            ):
                corrupted[channel] = [self._noisy(float(x), rng) for x in reading]
                affected += len(reading)
            else:
                corrupted[channel] = reading

        if affected == 0:
            return None
        return (
            corrupted,
            f"corrupted {len(payload)} sensor channel(s)",
            self.intensity,
            {"channels": list(payload.keys()), "values_affected": affected},
        )
