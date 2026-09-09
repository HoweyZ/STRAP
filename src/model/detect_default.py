"""GPU feature histograms for streaming-node drift detection."""

import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data


@torch.no_grad()
def get_feature(data, graph, args, model, adj):
    n_nodes = data.shape[1]
    samples = data[-288 * 7 - 1:-1].reshape(-1, args.x_len, n_nodes)
    x = torch.as_tensor(samples, dtype=torch.float32, device=args.device)
    x = x.transpose(1, 2).reshape(-1, args.x_len)
    features = model.feature(Data(x=x), adj)
    return features.reshape(-1, n_nodes, features.shape[-1]).permute(1, 0, 2)


def get_adj(year, args):
    array = np.load(osp.join(args.graph_path, f"{year}_adj.npz"))["x"]
    adj = torch.as_tensor(array, dtype=torch.float32, device=args.device)
    return adj / (adj.sum(dim=1, keepdim=True) + 1e-6)


def _histograms(samples, low, high, bins=10):
    """Uniform per-row histograms, including NumPy's constant-range convention."""
    constant = low == high
    low = low - constant * 0.5
    high = high + constant * 0.5
    indices = ((samples - low) / (high - low) * bins).long().clamp(0, bins - 1)
    counts = samples.new_zeros(samples.shape[0], bins)
    counts.scatter_add_(1, indices, torch.ones_like(samples))
    return counts / samples.shape[1]


def _original_scores(pre_data, cur_data, device):
    previous = torch.as_tensor(pre_data.T, dtype=torch.float64, device=device)
    current = torch.as_tensor(cur_data[:, :pre_data.shape[1]].T, dtype=torch.float64, device=device)
    low = torch.minimum(previous.amin(dim=1, keepdim=True), current.amin(dim=1, keepdim=True))
    high = torch.maximum(previous.amax(dim=1, keepdim=True), current.amax(dim=1, keepdim=True))
    p, q = _histograms(previous, low, high), _histograms(current, low, high)
    return torch.where(p > 0, p * (p.log() - q.log()), 0).sum(dim=1)


def _feature_scores(previous, current):
    n_nodes, _, n_features = previous.shape
    distributions = []
    for features in (previous, current[:n_nodes]):
        rows = features.permute(0, 2, 1).reshape(n_nodes * n_features, -1).double()
        distributions.append(_histograms(rows, rows.amin(dim=1, keepdim=True), rows.amax(dim=1, keepdim=True)))
    p, q = distributions
    midpoint = (p + q) / 2
    divergence = (
        0.5 * (torch.special.xlogy(p, p) + torch.special.xlogy(q, q))
        - torch.special.xlogy(midpoint, midpoint)
    ).sum(dim=1)
    return divergence.clamp_min(0).sqrt().reshape(n_nodes, n_features).sum(dim=1)


@torch.no_grad()
def score_func(pre_data, cur_data, args):
    scores = _original_scores(pre_data, cur_data, args.device)
    return scores.topk(args.topk).indices.cpu().numpy()


@torch.no_grad()
def influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph):
    if args.detect_strategy == "original":
        scores = _original_scores(pre_data[-288 * 7 - 1:-1], cur_data[-288 * 7 - 1:-1], args.device)
    elif args.detect_strategy == "feature":
        model.eval()
        previous = get_feature(pre_data, pre_graph, args, model, get_adj(args.year - 1, args))
        current = get_feature(cur_data, cur_graph, args, model, get_adj(args.year, args))
        scores = _feature_scores(previous, current)
    else:
        raise ValueError(f"Unknown detection strategy: {args.detect_strategy}")
    # The graph-selection callers consume one score per node on the CPU.
    return scores.cpu().numpy()
