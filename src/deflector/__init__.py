"""Deflector — grounded support deflection with a calibrated confidence gate."""

from .pipeline import Deflector, DeflectionResult
from .confidence import Band, Route

__version__ = "1.0.0"
__all__ = ["Deflector", "DeflectionResult", "Band", "Route"]
