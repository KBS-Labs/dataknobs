"""A seeded ``SensorDataGenerator`` is reproducible, whatever else runs.

``SensorDataGenerator(seed=42)`` seeded the **global** ``random`` module
and then drew from it, so its output depended on what every other
consumer of ``random`` in the process had drawn since. That is not a
theoretical hazard: it made
``test_sensor_dashboard_example.py::test_duplicate_handling`` fail
intermittently in full-suite runs and pass on its own, because
``generate_duplicate_readings`` produces no duplicates at all with
probability ``0.7 ** 20`` -- about one run in 1,250 -- once the seed is
no longer deciding the draws.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from sensor_dashboard import SensorDataGenerator

START = datetime(2025, 1, 17, 10, 0, 0)


def _readings(generator: SensorDataGenerator) -> list[tuple[float, float]]:
    """A fingerprint of one generator's draws."""
    return [
        (r.temperature, r.humidity)
        for r in generator.generate_normal_readings("s", "room_a", START, count=10)
    ]


def test_same_seed_reproduces_despite_an_interleaved_consumer() -> None:
    """An unrelated draw between construction and use must change nothing.

    This is the defect exactly: the generator seeded a global and drew
    from it later, so anything else drawing in between shifted its
    sequence. In a test suite that "anything else" is every other test.
    """
    baseline = _readings(SensorDataGenerator(seed=42))

    generator = SensorDataGenerator(seed=42)
    random.random()  # another test, another thread, anything at all
    interleaved = _readings(generator)

    assert interleaved == baseline


def test_seeded_generator_does_not_disturb_the_global_stream() -> None:
    """Constructing one must not reseed the caller's own randomness.

    The damage ran in both directions: seeding a global from a
    constructor also silently made every *other* consumer in the process
    reproducible, which is how a suite acquires order-dependent tests
    that nobody can explain.
    """
    random.seed(1234)
    expected = [random.random() for _ in range(3)]

    random.seed(1234)
    first = random.random()
    SensorDataGenerator(seed=42)
    rest = [random.random() for _ in range(2)]

    assert [first, *rest] == expected


def test_seed_zero_seeds() -> None:
    """``seed=0`` is a seed. ``if seed:`` read it as "no seed given"."""
    assert _readings(SensorDataGenerator(seed=0)) == _readings(SensorDataGenerator(seed=0))


def test_different_seeds_differ() -> None:
    """The false-positive guard: reproducibility must not mean constancy."""
    assert _readings(SensorDataGenerator(seed=1)) != _readings(SensorDataGenerator(seed=2))


def test_unseeded_generators_are_independent() -> None:
    """Two unseeded generators still draw their own values."""
    assert _readings(SensorDataGenerator()) != _readings(SensorDataGenerator())
