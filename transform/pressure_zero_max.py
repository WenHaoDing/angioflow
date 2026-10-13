"""Re-reference interior pressure to each case's own maximum: p -> p - p.max().

Fluent's pressure field carries a large, physically arbitrary offset (this dataset's
convention is ~1e5 Pa, near-atmospheric) that varies slightly case to case and frame
to frame for reasons unrelated to the flow field's actual shape -- only RELATIVE
pressure is physically meaningful, and some regions (recirculation, the aneurysm
dome) can legitimately sit below whatever reference Fluent reports, i.e. "negative
pressure". Re-referencing to the case's own maximum (rather than a shared global
mean, the previous convention) gives every case a physically motivated zero point --
its own highest-pressure location, typically a stagnation point -- instead of an
arbitrary statistical constant. After this transform every point's pressure is <= 0.

Must run BEFORE any subsetting (crop / subsample): the max is taken over the FULL,
untouched interior pool for this (case, frame) sample, so the reference stays a
fixed, reproducible property of the case regardless of which points a later random
crop/subsample happens to keep that epoch. Runs on the raw (wall, interior) pair,
operating per single graph (pre-batch), like AneurysmCrop.
"""
from __future__ import annotations


class PressureZeroMax:
    def __call__(self, data):
        wall, interior = data
        interior.press = interior.press - interior.press.max()
        return wall, interior
