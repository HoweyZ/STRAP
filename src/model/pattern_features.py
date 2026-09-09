"""STRAP graph partitions and key/value construction.

Discrete graph algorithms run once per build. Numerical descriptors and backbone
values use the input device; no feature tensors are copied to NumPy.
"""

from collections import Counter, defaultdict

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering
import torch
from torch import nn
from torch.nn import functional as F


PATTERN_TYPES = ("spatial", "temporal", "spatiotemporal")


def fit_features(x, width):
    """Original STRAP descriptor convention: truncate or zero-pad the last axis."""
    return F.pad(x[..., :width], (0, max(0, width - x.shape[-1])))


class WaveletFeatures(nn.Module):
    """Batched db4 DWT with half-sample symmetric boundary extension."""

    def __init__(self):
        super().__init__()
        low = torch.tensor([
            -0.010597401785069032, 0.0328830116668852, 0.030841381835560764,
            -0.18703481171909309, -0.027983769416859854, 0.6308807679298587,
            0.7148465705529154, 0.2303778133088964,
        ])
        high = low.flip(0) * torch.tensor([-1., 1., -1., 1., -1., 1., -1., 1.])
        self.register_buffer("filters", torch.stack((low, high)).flip(-1).unsqueeze(1))

    def coefficients(self, signals, level):
        approximation = signals.reshape(-1, signals.shape[-1])
        details = []
        for _ in range(level):
            length = approximation.shape[-1]
            positions = torch.arange(-6, length + 6 + length % 2, device=signals.device)
            positions = positions.remainder(2 * length)
            positions = torch.where(positions < length, positions, 2 * length - 1 - positions)
            coefficients = F.conv1d(
                approximation[:, positions].unsqueeze(1), self.filters, stride=2
            )
            approximation = coefficients[:, 0]
            details.append(coefficients[:, 1])
        return [approximation, *reversed(details)]

    def forward(self, signals, level=4, moments=False):
        coefficients = self.coefficients(signals, level)
        if moments:
            return torch.cat([
                torch.stack((c.mean(-1), c.std(-1, unbiased=False), c.amax(-1), c.amin(-1)), -1)
                for c in coefficients
            ], -1)
        return torch.stack([c.abs().mean(-1) for c in coefficients], -1)


class FormanRicciCurvature:
    """Vectorized versions of the node/edge formulas in the original STRAP."""

    @staticmethod
    def compute(adj):
        adjacency = ((adj > 0) | (adj.t() > 0)).to(adj.dtype)
        neighbors = adjacency.sum(1)
        degree = neighbors + adjacency.diagonal()  # NetworkX counts self-loops twice.
        nodes = 1 - degree * (adjacency @ degree) / 2 + degree * neighbors
        nodes = torch.where(neighbors > 0, nodes, torch.zeros_like(nodes))
        edges = adjacency.triu().nonzero()
        # Only edge-wise common-neighbor counts are needed, not a dense A squared.
        common = torch.cat([
            (adjacency[pair[:, 0]] * adjacency[pair[:, 1]]).sum(1)
            for pair in edges.split(1024)
        ])
        curvature = 2.5 - neighbors[edges[:, 0]] - neighbors[edges[:, 1]] + common
        return nodes, edges, curvature

    @classmethod
    def identify_patterns(cls, adj):
        nodes, edges, curvature = cls.compute(adj)
        return {
            "negative": (nodes < -1e-6).nonzero().flatten(),
            "zero": (nodes.abs() <= 1e-6).nonzero().flatten(),
            "positive": (nodes > 1e-6).nonzero().flatten(),
        }, {
            "high_flow": edges[curvature > 0.5],
            "fluctuation": edges[curvature.abs() <= 0.5],
            "bottleneck": edges[curvature < -0.5],
        }


class GraphLayout:
    """Share one CPU graph, partitions and uploaded indices across all patterns."""

    def __init__(self, adj, k_hop, n_clusters, max_neighbors, max_cluster_nodes, enabled):
        adjacency = ((adj + adj.t()) / 2).detach().cpu().numpy()
        graph = nx.from_numpy_array(adjacency)
        count = len(graph)
        components = [sorted(c) for c in nx.connected_components(graph)]
        self.groups = defaultdict(list)
        groups = {}

        def add(source, nodes):
            nodes = tuple(nodes)
            groups.setdefault(nodes, len(groups))
            self.groups[source].append(groups[nodes])

        def partition(source, nodes, minimum):
            for start in range(0, len(nodes), max_cluster_nodes):
                part = nodes[start:start + max_cluster_nodes]
                if len(part) >= minimum:
                    add(source, part)

        if enabled["spatial"]:
            by_degree = defaultdict(list)
            for node, degree in graph.degree():
                by_degree[degree].append(node)
            for nodes in by_degree.values():
                partition("degree", nodes, 5)
            for component in components:
                if len(component) >= 5:
                    communities = nx.community.greedy_modularity_communities(graph.subgraph(component))
                    for community in communities:
                        partition("community", sorted(community), 5)
            for node in graph:
                neighbors = sorted(set(graph.neighbors(node)) - {node})[:max_neighbors]
                add("curvature_node", [*neighbors, node])

        if enabled["temporal"]:
            for node in graph:
                distances = nx.single_source_shortest_path_length(graph, node, cutoff=k_hop)
                neighbors = sorted((n for n in distances if n != node), key=lambda n: (distances[n], n))
                add("k_hop", [*neighbors[:max_neighbors], node])

        if enabled["spatiotemporal"]:
            clusters = min(count, n_clusters if n_clusters is not None else max(2, min(5, count // 200)))
            # k == N has singleton clusters by definition and needs no eigensolver.
            if clusters == count:
                labels = np.arange(count)
            else:
                labels = SpectralClustering(
                    n_clusters=clusters, affinity="precomputed", random_state=42
                ).fit_predict(adjacency)
            for cluster in range(clusters):
                partition("spectral", np.flatnonzero(labels == cluster).tolist(), 3)

        indices, offsets, structure = [], [0], []
        for nodes in groups:
            indices.extend(nodes)
            offsets.append(len(indices))
            subgraph = graph.subgraph(nodes)
            largest = subgraph.subgraph(max(nx.connected_components(subgraph), key=len))
            structure.append((nx.average_clustering(subgraph), nx.density(subgraph),
                              nx.average_shortest_path_length(largest)))
        packed = torch.tensor(indices, device=adj.device, dtype=torch.long)
        self.nodes = [packed[start:end] for start, end in zip(offsets, offsets[1:])]
        self.structure = torch.tensor(structure, device=adj.device, dtype=adj.dtype).reshape(-1, 3)

        topology = np.zeros((count, 16), dtype=adjacency.dtype)
        degree = dict(graph.degree())
        centrality = nx.degree_centrality(graph)
        clustering = nx.clustering(graph)
        for component in components:
            subgraph = graph.subgraph(component)
            closeness = nx.closeness_centrality(subgraph)
            betweenness = nx.betweenness_centrality(
                subgraph, k=min(100, len(component)), seed=42
            )
            matrix = nx.to_scipy_sparse_array(subgraph, nodelist=component, weight=None, dtype=float)
            if len(component) <= 2:
                eigenvector = np.linalg.eigh(matrix.toarray())[1][:, -1]
            else:
                eigenvector = eigsh(matrix, k=1, which="LA", v0=np.ones(len(component)))[1][:, 0]
            for node, eigen in zip(component, np.abs(eigenvector)):
                neighbors = [degree[n] for n in graph.neighbors(node)]
                topology[node, :6] = (centrality[node], clustering[node], closeness[node],
                                       betweenness[node], eigen, len(neighbors) / count)
                if neighbors:
                    topology[node, 6:8] = (np.mean(neighbors) / count, np.std(neighbors) / count)
        self.topology = torch.from_numpy(topology).to(adj.device)


class PatternExtractor(nn.Module):
    def __init__(self, args, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.k_hop = int(getattr(args, "temporal_k_hop", 2))
        self.n_clusters = getattr(args, "spatial_clusters", None)
        self.max_neighbors = int(getattr(args, "max_neighbors", 20))
        self.max_cluster_nodes = int(getattr(args, "max_cluster_nodes", 100))
        self.time_window = int(getattr(args, "time_window", 12))
        self.overlap = int(getattr(args, "time_overlap", 6))
        self.cross_count = int(getattr(args, "cross_pattern_count", 2000))
        self.curvature_count = int(getattr(args, "curvature_pattern_count", 2000))
        if min(self.k_hop, self.max_neighbors, self.max_cluster_nodes, self.cross_count, self.curvature_count) < 1:
            raise ValueError("Pattern construction sizes must be positive")
        if not 0 <= self.overlap < self.time_window:
            raise ValueError("Require 0 <= time_overlap < time_window")
        if self.n_clusters is not None and self.n_clusters < 1:
            raise ValueError("spatial_clusters must be positive")
        self.wavelet = WaveletFeatures()

    def _value(self, features):
        return F.normalize(fit_features(features, self.feature_dim), dim=-1)

    def process_subgraph(self, features, adj, structure, temporal, backbone):
        degree = adj.sum(1)
        descriptors = [
            features.mean(0).mean(0), features.std(0, unbiased=False).mean(0),
            features.amax(0).mean(0), features.amin(0).mean(0),
            torch.stack((degree.mean(), degree.std(unbiased=False), degree.amax(), degree.amin())),
            structure,
        ]
        if temporal:
            signals = features[:, :, :3].permute(1, 2, 0)
            descriptors.append(self.wavelet(signals, level=2, moments=True).mean(0))
        key = torch.stack([F.normalize(fit_features(d, self.feature_dim), dim=-1) for d in descriptors]).mean(0)
        value = self._value(backbone(features, adj).mean((0, 1)))
        return key, value

    def _cross(self, spatial, temporal):
        # Chunk the pair search and outer products; no scalar indices leave the device.
        spatial_keys = F.normalize(spatial["keys"], dim=-1)
        temporal_keys = F.normalize(temporal["keys"], dim=-1)
        scores = spatial_keys.new_empty(0)
        pairs = torch.empty(0, device=scores.device, dtype=torch.long)
        for start in range(0, len(spatial["keys"]), 256):
            for time_start in range(0, len(temporal_keys), 4096):
                chunk = temporal_keys[time_start:time_start + 4096]
                affinity = spatial_keys[start:start + 256] @ chunk.t()
                score, pair = affinity.flatten().topk(min(self.cross_count, affinity.numel()))
                pair = (pair // len(chunk) + start) * len(temporal_keys) + pair % len(chunk) + time_start
                candidates = torch.cat((scores, score))
                scores, selected = candidates.topk(min(self.cross_count, len(candidates)))
                pairs = torch.cat((pairs, pair))[selected]
        spatial_index, temporal_index = pairs // len(temporal["keys"]), pairs % len(temporal["keys"])
        result = {}
        for name in ("keys", "values"):
            pooled = []
            for left, right in zip(spatial_index.split(256), temporal_index.split(256)):
                outer = spatial[name][left].unsqueeze(2) * temporal[name][right].unsqueeze(1)
                pooled.append(F.normalize(F.adaptive_avg_pool1d(outer.flatten(1).unsqueeze(1), self.feature_dim).squeeze(1), dim=-1))
            result[name] = torch.cat(pooled)
        result["topology"] = (spatial["topology"][spatial_index] + temporal["topology"][temporal_index]) / 2
        return result

    @torch.no_grad()
    def forward(self, x, adj, backbone, enabled):
        """x is chronological [samples, nodes, input channels], already on device."""
        layout = GraphLayout(adj, self.k_hop, self.n_clusters, self.max_neighbors, self.max_cluster_nodes, enabled)
        records = {kind: {name: [] for name in ("keys", "values", "topology")} for kind in PATTERN_TYPES}
        counts = {kind: Counter() for kind in PATTERN_TYPES}

        def append(kind, source, keys, values, topology):
            for name, tensor in zip(("keys", "values", "topology"), (keys, values, topology)):
                records[kind][name].append(tensor.reshape(-1, tensor.shape[-1]))
            counts[kind][source] += keys.reshape(-1, self.feature_dim).shape[0]

        for kind, sources in (("spatial", ("degree", "community")), ("temporal", ("k_hop",)),
                              ("spatiotemporal", ("spectral",))):
            for source in sources:
                for group in layout.groups[source]:
                    nodes = layout.nodes[group]
                    sub_adj = adj[nodes[:, None], nodes]
                    windows = range(0, len(x), self.time_window - self.overlap) if source == "spectral" else (0,)
                    if source == "spectral":
                        sub_adj = (sub_adj + sub_adj.t()) / 2
                    for start in windows:
                        features = x[start:start + self.time_window, nodes] if source == "spectral" else x[:, nodes]
                        key, value = self.process_subgraph(features, sub_adj, layout.structure[group], kind != "spatial", backbone)
                        append(kind, source, key, value, layout.topology[nodes].mean(0))

        if enabled["temporal"]:
            # The original level-4 descriptor operates on each node's input window.
            # Values must encode that actual input, not the five wavelet statistics.
            wavelet = fit_features(self.wavelet(x), self.feature_dim)
            value = backbone(x.reshape(-1, 1, x.shape[-1]), x.new_ones(1, 1)).squeeze(1)
            append("temporal", "wavelet", wavelet, fit_features(value, self.feature_dim), layout.topology.repeat(len(x), 1))

        if enabled["spatial"]:
            node_groups, edge_groups = FormanRicciCurvature.identify_patterns(adj)
            center_values = []
            for group in layout.groups["curvature_node"]:
                nodes = layout.nodes[group]
                center_values.append(backbone(x[:, nodes], adj[nodes[:, None], nodes])[:, -1].mean(0))
            center_values = torch.stack(center_values)
            degree = adj.sum(1, keepdim=True) / adj.shape[0]
            for category, (source, nodes) in enumerate(node_groups.items()):
                nodes = nodes[torch.randperm(len(nodes), device=x.device)[:self.curvature_count]]
                indicator = F.one_hot(torch.full_like(nodes, category), 3).to(x.dtype)
                key = torch.cat((indicator, degree[nodes], layout.topology[nodes]), -1)
                append("spatial", "curvature_" + source, fit_features(key, self.feature_dim), fit_features(center_values[nodes], self.feature_dim), layout.topology[nodes])
            for category, (source, edges) in enumerate(edge_groups.items()):
                edges = edges[torch.randperm(len(edges), device=x.device)[:self.curvature_count]]
                topology = layout.topology[edges].mean(1)
                indicator = F.one_hot(torch.full((len(edges),), category, device=x.device), 3).to(x.dtype)
                key = torch.cat((indicator, degree[edges].mean(1), topology), -1)
                append("spatial", "curvature_" + source, fit_features(key, self.feature_dim), fit_features(center_values[edges].mean(1), self.feature_dim), topology)

        libraries = {}
        for kind in PATTERN_TYPES:
            libraries[kind] = {
                name: torch.cat(parts) if parts else x.new_empty(0, 16 if name == "topology" else self.feature_dim)
                for name, parts in records[kind].items()
            }
        if enabled["spatiotemporal"] and enabled["spatial"] and enabled["temporal"]:
            cross = self._cross(libraries["spatial"], libraries["temporal"])
            for name in cross:
                libraries["spatiotemporal"][name] = torch.cat((libraries["spatiotemporal"][name], cross[name]))
            counts["spatiotemporal"]["cross"] = len(cross["keys"])
        return libraries, {kind: dict(counts[kind]) for kind in PATTERN_TYPES}
