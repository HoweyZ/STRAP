"""STKEC entry points for the shared trainer."""

from src.trainer import engine


def train(inputs, args):
    return engine.train(inputs, args, with_attention=True)


def test_model(model, args, testset, pin_memory):
    return engine.test_model(model, args, testset, pin_memory, with_attention=True)
