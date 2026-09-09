"""Masked forecasting metrics accumulated on the prediction device."""

import math

import torch


def _mask(y_true, null_val):
    return ~torch.isnan(y_true) if math.isnan(null_val) else y_true != null_val


def masked_mae(y_true, y_pred, null_val=0):
    mask = _mask(y_true, null_val)
    errors = torch.where(mask, (y_true - y_pred).abs(), 0)
    # An entirely masked batch contributes zero, matching the metric definition.
    return errors.nan_to_num().sum() / mask.sum().clamp_min(1)


class ForecastMetrics:
    """Sufficient statistics per forecast step; no prediction history is retained."""

    def __init__(self, horizon, device, null_val=0):
        self.null_val = null_val
        # valid count, absolute error, squared error, absolute percentage error
        self.totals = torch.zeros(4, horizon, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, y_true, y_pred):
        mask = _mask(y_true, self.null_val)
        error = y_true - y_pred
        errors = torch.stack((error.abs(), error.square(), (error / y_true).abs()))
        errors = torch.where(mask.unsqueeze(0), errors, 0).nan_to_num()
        dims = tuple(range(y_true.ndim - 1))
        self.totals[0].add_(mask.sum(dim=dims))
        self.totals[1:].add_(errors.sum(dim=tuple(d + 1 for d in dims), dtype=torch.float64))

    def compute(self):
        """Return [horizon, (MAE, RMSE, MAPE)] for cumulative horizons 1..H."""
        totals = self.totals.cumsum(dim=1)
        means = totals[1:] / totals[0].clamp_min(1)
        return torch.stack((means[0], means[1].sqrt(), means[2] * 100), dim=1)


def log_metrics(metrics, args):
    # Only the final H x 3 summary crosses to the host for logging and reporting.
    scores = metrics.compute().cpu().tolist()
    args.logger.info("[*] year %s, testing", args.year)
    for horizon in (3, 6, 12):
        mae, rmse, mape = scores[horizon - 1]
        args.logger.info("T:%d\tMAE\t%.4f\tRMSE\t%.4f\tMAPE\t%.4f", horizon, mae, rmse, mape)
        for name, value in zip((" MAE", "RMSE", "MAPE"), (mae, rmse, mape)):
            args.result[str(horizon)][name][args.year] = value
    averages = [sum(column) / len(scores) for column in zip(*scores)]
    for name, value in zip((" MAE", "RMSE", "MAPE"), averages):
        args.result["Avg"][name][args.year] = value
    args.logger.info("T:Avg\tMAE\t%.4f\tRMSE\t%.4f\tMAPE\t%.4f", *averages)


def cal_metric(ground_truth, prediction, args):
    metrics = ForecastMetrics(prediction.shape[-1], prediction.device)
    metrics.update(ground_truth, prediction)
    log_metrics(metrics, args)
