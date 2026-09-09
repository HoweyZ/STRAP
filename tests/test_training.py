import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from src.model.model import RAP_Model
from src.model.ewc import EWC
from src.trainer.engine import train
from utils.common_tools import load_best_model


class AttentionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.forecast = nn.Linear(12, 12)
        self.attention = nn.Linear(12, 3)

    def forward(self, data, adj):
        return self.forecast(data.x), self.attention(data.x).softmax(dim=1)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.args = SimpleNamespace(
            path=self.directory.name, year=0, begin_year=0, strategy="retrain",
            dropout=0., device="cpu", y_len=12, loss="mse", epoch=2, batch_size=2,
            num_workers=0, lr=.01, adj=torch.eye(3), init=True, ewc=False,
            method="RAP", methods={"RAP": RAP_Model}, train=1, load_first_year=False,
            gcn={"in_channel": 12, "out_channel": 12, "hidden_channel": 8},
            tcn={"in_channel": 1, "out_channel": 1, "kernel_size": 3, "dilation": 1},
            logger=logging.getLogger("strap-test"), graph_size=3,
            result={key: {metric: {} for metric in (" MAE", "RMSE", "MAPE")}
                    for key in ("3", "6", "12", "Avg")},
        )
        rng = np.random.default_rng(3)
        self.inputs = {f"{split}_{xy}": rng.uniform(.2, 1., (5, 12, 3)).astype(np.float32)
                       for split in ("train", "val", "test") for xy in ("x", "y")}

    def test_two_year_rap_training_and_checkpoint_year(self):
        train(self.inputs, self.args)
        self.args.year = 1
        loaded, loss = load_best_model(self.args)
        self.assertEqual(loaded.current_year, 0)
        self.assertEqual(loaded.strap.current_year, 0)
        self.assertEqual(float(loss), min(float(p.stem) for p in (Path(self.args.path) / "0").glob("*.pkl")))
        train(self.inputs, self.args)
        self.assertTrue((Path(self.args.path) / "pattern_libraries/1_spatiotemporal.pkl").is_file())
        self.assertTrue(np.isfinite(self.args.result["Avg"][" MAE"][1]))

    def test_attention_training_retains_auxiliary_gradients(self):
        self.args.method = "STKEC"
        models = []
        def factory(args):
            model = AttentionModel(args)
            models.append(model)
            return model
        self.args.methods = {"STKEC": factory}
        self.args.attention = np.eye(3, dtype=np.float32)
        train(self.inputs, self.args, with_attention=True)
        self.assertGreater(models[0].attention.weight.grad.norm().item(), 0)
        self.assertTrue(np.isfinite(self.args.result["Avg"][" MAE"][0]))

    def test_incremental_attention_ewc_training(self):
        from src.model.model import STKEC_Model
        self.args.method = "STKEC"
        self.args.methods = {"STKEC": STKEC_Model}
        self.args.cluster = 3
        self.args.attention = np.eye(3, dtype=np.float32)
        train(self.inputs, self.args, with_attention=True)
        self.args.year = 1
        self.args.strategy = "incremental"
        self.args.subgraph = torch.tensor([0, 2])
        self.args.subgraph_edge_index = torch.tensor([[0], [1]])
        self.args.mapping = torch.tensor([1])
        self.args.ewc = True
        self.args.ewc_lambda, self.args.ewc_strategy = .1, "ewc"
        train(self.inputs, self.args, with_attention=True)
        torch.testing.assert_close(self.args.sub_adj, torch.tensor([[0., 1.], [1., 0.]]), atol=2e-6, rtol=0)
        self.assertTrue(np.isfinite(self.args.result["Avg"][" MAE"][1]))

    def test_checkpoint_selection_uses_numeric_loss(self):
        directory = Path(self.args.path) / "0"
        directory.mkdir()
        state = {"model_state_dict": RAP_Model(self.args).state_dict()}
        for loss in ("10.0", "2.0"):
            torch.save(state, directory / f"{loss}.pkl")
        self.args.year = 1
        _, loss = load_best_model(self.args)
        self.assertEqual(loss, "2.0")

    def test_ewc_fisher_and_consolidation_gradients(self):
        class Predictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.))
            def forward(self, data, adj):
                return data.x * self.weight
        model = Predictor()
        ewc = EWC(model, torch.eye(1), ewc_lambda=2.)
        data = Data(x=torch.tensor([[3.]]), y=torch.tensor([[1.]]))
        ewc.register_ewc_params([data], nn.functional.mse_loss, "cpu")
        self.assertEqual(ewc.weight_estimated_fisher.item(), 900.)
        with torch.no_grad():
            model.weight.add_(1.)
        loss = ewc.compute_consolidation_loss()
        self.assertEqual(loss.item(), 900.)
        loss.backward()
        self.assertEqual(model.weight.grad.item(), 1800.)


if __name__ == "__main__":
    unittest.main()
