import logging
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from torch.nn import functional as F

from src.model.STRAP import STRAP
from src.model.model import RAP_Model, PECPM_Model
from src.model.detect_default import _feature_scores, _original_scores
from utils.metric import ForecastMetrics, masked_mae


def make_args(path, **kwargs):
    args = dict(
        path=path, year=0, begin_year=0, dropout=0., y_len=12,
        gcn={"in_channel": 12, "out_channel": 12, "hidden_channel": 8},
        tcn={"in_channel": 1, "out_channel": 1, "kernel_size": 3, "dilation": 1},
        k_neighbors=4, max_patterns=9, retrieval_batch_size=3,
        logger=logging.getLogger("strap-test"), attention_weight=3,
    )
    args.update(kwargs)
    return SimpleNamespace(**args)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.args = make_args(self.directory.name)

    def test_chunked_outputs_and_gradients_match_dense_reference(self):
        model = STRAP(self.args).double()
        model.patterns = F.normalize(torch.randn(9, 8, dtype=torch.float64), dim=-1)
        model.values = torch.randn(9, 8, dtype=torch.float64)
        query = torch.randn(11, 8, dtype=torch.float64, requires_grad=True)
        scores, indices = (F.normalize(query, dim=-1) @ model.patterns.t()).topk(4, dim=1)
        expected = (scores.softmax(dim=1).unsqueeze(-1) * model.values[indices]).sum(1)
        actual = model._retrieve(query)
        torch.testing.assert_close(actual, expected)
        expected_grad, = torch.autograd.grad(expected.square().sum(), query, retain_graph=True)
        actual_grad, = torch.autograd.grad(actual.square().sum(), query)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_library_roundtrip_and_same_year_do_not_reload(self):
        model = STRAP(self.args)
        x = SimpleNamespace(x=torch.randn(30, 12))
        model.extract_patterns(x, year=0)
        self.assertEqual(model.patterns.shape, (9, 8))
        self.assertFalse(model.patterns.requires_grad)
        restored = STRAP(self.args)
        self.assertTrue(restored.switch_to_year(0))
        torch.testing.assert_close(restored.patterns, model.patterns)
        torch.testing.assert_close(restored.values, model.values)
        pointer = restored.patterns.data_ptr()
        with patch.object(restored.pattern_manager, "get_library_for_year", side_effect=AssertionError):
            self.assertTrue(restored.switch_to_year(0))
        self.assertEqual(pointer, restored.patterns.data_ptr())
        restored.double()
        self.assertEqual(restored.patterns.dtype, torch.float64)
        self.assertEqual(restored.values.dtype, torch.float64)
        self.assertNotIn("patterns", restored.state_dict())
        self.assertIn("projector.weight", restored.state_dict())

    def test_uninitialized_and_missing_year_fail_without_bypass(self):
        model = STRAP(self.args)
        with self.assertRaises(RuntimeError):
            model(torch.randn(3, 8))
        model.extract_patterns(SimpleNamespace(x=torch.randn(5, 12)), year=0)
        self.assertFalse(model.switch_to_year(1))
        self.assertEqual(model.patterns.shape[0], 0)
        with self.assertRaises(RuntimeError):
            model(torch.randn(3, 8))
        rap = RAP_Model(self.args).eval()
        rap.set_year(1)
        with self.assertRaises(FileNotFoundError):
            rap(SimpleNamespace(x=torch.randn(10, 12)), torch.eye(5))

    def test_checkpoint_reload_and_rap_gradients(self):
        model = RAP_Model(self.args)
        data, adj = SimpleNamespace(x=torch.randn(10, 12)), torch.eye(5)
        output = model(data, adj)
        output.square().sum().backward()
        self.assertGreater(model.strap_adapter.weight.grad.norm().item(), 0)
        restored = RAP_Model(self.args)
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        restored.eval()
        torch.testing.assert_close(restored(data, adj), model(data, adj))
        # Neither the prepared forward path nor its error handling may copy or bypass.
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError), \
             patch.object(torch.Tensor, "to", side_effect=AssertionError):
            restored(data, adj)
        with patch.object(restored.strap, "forward", side_effect=ValueError("retrieval error")):
            with self.assertRaisesRegex(ValueError, "retrieval error"):
                restored(data, adj)

    def test_small_library_pattern_mode_and_disabled_strap(self):
        model = STRAP(make_args(self.directory.name, return_pattern_or_value="pattern"))
        model.extract_patterns(SimpleNamespace(x=torch.randn(2, 12)), year=0)
        query = torch.randn(3, 8)
        torch.testing.assert_close(model(query).norm(dim=1), torch.ones(3))
        rap = RAP_Model(make_args(self.directory.name, use_strap=False)).eval()
        self.assertEqual(rap(SimpleNamespace(x=torch.randn(10, 12)), torch.eye(5)).shape, (10, 12))

    def test_pecpm_matches_cosine_scores_without_host_transfers(self):
        model = PECPM_Model(self.args)
        history, current = torch.randn(6, 12), torch.randn(4, 12)
        model.pattern_matching(history)
        expected = (F.normalize(current, dim=1) @ F.normalize(history, dim=1).t()).topk(3, dim=1).values.mean(1, keepdim=True)
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError):
            actual = model.pattern_matching(current)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)
        model.double()
        self.assertEqual(model.historical_patterns.dtype, torch.float64)

    def test_dynamic_adapters_follow_model_device_and_dtype(self):
        from src.model.model import STLora_Model, STAdapter_Model
        for model_type in (STLora_Model, STAdapter_Model):
            model = model_type(self.args).double()
            if model_type is STLora_Model:
                model.add_lora_layer()
            else:
                model.add_adapter_group()
            output = model(SimpleNamespace(x=torch.randn(10, 12, dtype=torch.float64)),
                           torch.eye(5, dtype=torch.float64))
            output.sum().backward()
            self.assertEqual(output.dtype, torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for residency/profiler verification")
    def test_cuda_retrieval_has_no_host_transfers(self):
        model = STRAP(self.args).cuda()
        model.extract_patterns(SimpleNamespace(x=torch.randn(30, 12, device="cuda")), year=0)
        query = torch.randn(11, 8, device="cuda", requires_grad=True)
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError), \
             patch.object(torch.Tensor, "numpy", side_effect=AssertionError), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError), \
             patch.object(torch.Tensor, "to", side_effect=AssertionError):
            model(query).sum().backward()
        self.assertEqual(model.patterns.device, query.device)
        self.assertEqual(model.values.device, query.device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            model(query)
        names = [event.name.lower() for event in profiler.events()]
        self.assertFalse(any("dtoh" in name or "htod" in name for name in names))


class MetricTests(unittest.TestCase):
    def test_streamed_unequal_batches_match_numpy_prefix_metrics(self):
        rng = np.random.default_rng(5)
        truth = rng.uniform(0.5, 3., (9, 4, 12)).astype(np.float32)
        prediction = truth + rng.normal(0, .2, truth.shape).astype(np.float32)
        truth[:4, :, 3:5] = 0
        stats = ForecastMetrics(12, "cpu")
        for start, end in ((0, 4), (4, 8), (8, 9)):
            stats.update(torch.from_numpy(truth[start:end]), torch.from_numpy(prediction[start:end]))
        expected = []
        for horizon in range(1, 13):
            target, pred = truth[..., :horizon], prediction[..., :horizon]
            valid = target != 0
            error = target[valid] - pred[valid]
            expected.append([np.abs(error).mean(), np.sqrt(np.square(error).mean()),
                             np.abs(error / target[valid]).mean() * 100])
        np.testing.assert_allclose(stats.compute().numpy(), expected, rtol=2e-6, atol=1e-7)
        self.assertEqual(stats.totals.numel(), 48)

    def test_masks_nan_and_empty_valid_sets(self):
        truth = torch.tensor([[[float("nan"), 2., 0.]]])
        pred = torch.tensor([[[4., 3., 9.]]])
        self.assertEqual(masked_mae(truth, pred, float("nan")).item(), 5.)
        stats = ForecastMetrics(3, "cpu", null_val=float("nan"))
        stats.update(truth, pred)
        self.assertEqual(stats.compute()[0, 0].item(), 0.)
        zeros = torch.zeros(2, 3, 12)
        self.assertEqual(masked_mae(zeros, torch.ones_like(zeros)).item(), 0.)
        stats = ForecastMetrics(12, "cpu")
        stats.update(zeros, torch.ones_like(zeros))
        torch.testing.assert_close(stats.compute(), torch.zeros(12, 3, dtype=torch.float64))


class DriftTests(unittest.TestCase):
    def test_original_histogram_kl_matches_scipy(self):
        rng = np.random.default_rng(6)
        previous, current = rng.normal(size=(300, 5)), rng.normal(.1, size=(400, 7))
        expected = []
        for i in range(5):
            limits = (min(previous[:, i].min(), current[:, i].min()),
                      max(previous[:, i].max(), current[:, i].max()))
            expected.append(entropy(np.histogram(previous[:, i], 10, limits)[0],
                                    np.histogram(current[:, i], 10, limits)[0]))
        np.testing.assert_allclose(_original_scores(previous, current, "cpu").numpy(), expected, atol=1e-12)

    def test_feature_histograms_match_scipy_js(self):
        torch.manual_seed(2)
        previous, current = torch.randn(4, 170, 3), torch.randn(6, 190, 3)
        expected = []
        for i in range(4):
            score = 0
            for j in range(3):
                histograms = []
                for values in (previous[i, :, j].numpy(), current[i, :, j].numpy()):
                    normalized = (values - values.min()) / (values.max() - values.min())
                    histograms.append(np.histogram(normalized, 10, (0, 1))[0])
                score += jensenshannon(*histograms)
            expected.append(score)
        np.testing.assert_allclose(_feature_scores(previous, current).numpy(), expected, atol=1e-12)
        constant = torch.ones(4, 170, 3)
        torch.testing.assert_close(_feature_scores(constant, constant), torch.zeros(4, dtype=torch.float64))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
