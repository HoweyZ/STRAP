"""Prediction-only entry points for the shared trainer."""

from src.trainer.engine import train, test_model

__all__ = ["train", "test_model"]
