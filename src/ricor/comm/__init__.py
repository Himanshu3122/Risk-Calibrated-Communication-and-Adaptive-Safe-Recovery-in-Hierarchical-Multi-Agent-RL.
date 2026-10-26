"""Communication channel and fault taxonomy."""

from ricor.comm.channel import Channel, Delivery
from ricor.comm.faults import COMPOUND, HELD_OUT, ISOLATED, FaultSpec, compose, evaluation_spec, sample_training_spec

__all__ = [
    "Channel",
    "Delivery",
    "FaultSpec",
    "compose",
    "evaluation_spec",
    "sample_training_spec",
    "ISOLATED",
    "COMPOUND",
    "HELD_OUT",
]
