"""Compatibility shim: STRAP implementation moved to model.py."""

from model.model import STRAP, PatternLibraryManager, FormanRicciCurvature, RandomProjection

__all__ = ["STRAP", "PatternLibraryManager", "FormanRicciCurvature", "RandomProjection"]
