"""Three-library STRAP: paired history, device retrieval and feature fusion."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .pattern_features import PATTERN_TYPES, PatternExtractor


class PatternLibraryManager:
    """Version a complete year's three libraries together; disk is the CPU boundary."""

    def __init__(self, args):
        self.base_dir = Path(args.path) / "pattern_libraries" / "strap_full_v1"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.base_dir / "metadata.json"
        self.metadata = json.loads(self.metadata_path.read_text()) if self.metadata_path.exists() else {}

    def get_library_for_year(self, year, device):
        versions = self.metadata.get(str(year), [])
        if not versions:
            return None
        return torch.load(self.base_dir / versions[-1]["path"], map_location=device, weights_only=True)

    def update_library(self, year, libraries, params, source_counts):
        versions = self.metadata.setdefault(str(year), [])
        version = len(versions) + 1
        filename = f"{year}_v{version}.pt"
        payload = {
            "libraries": {
                kind: {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                       for name, value in library.items()}
                for kind, library in libraries.items()
            },
            "params": params,
            "source_counts": source_counts,
        }
        torch.save(payload, self.base_dir / filename)
        versions.append({"version": version, "path": filename, "source_counts": source_counts, "params": params})
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2) + "\n")


class RandomProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        generator = torch.Generator().manual_seed(42)
        weight = torch.randint(2, (input_dim, output_dim), generator=generator).float()
        self.register_buffer("weight", (weight * 2 - 1) / output_dim ** 0.5)

    def forward(self, x):
        return x @ self.weight


class PatternBank(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        for name, width in (("keys", feature_dim), ("values", feature_dim), ("topology", 16)):
            self.register_buffer(name, torch.empty(0, width), persistent=False)

    def assign(self, library):
        for name in ("keys", "values", "topology"):
            setattr(self, name, library[name])


def sample_pairs(library, count, weights):
    """One index selection for every field: historical keys never lose their values."""
    indices = torch.multinomial(weights, count, replacement=False)
    return {name: values[indices] for name, values in library.items()}


class STRAP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_dim = args.gcn["hidden_channel"]
        self.k_neighbors = int(getattr(args, "k_neighbors", 50))
        self.history_ratio = float(getattr(args, "history_ratio", 0.3))
        self.begin_year = int(args.begin_year)
        self.history_limit = int(getattr(args, "history_limit", 10000))
        self.retrieval_batch_size = int(getattr(args, "retrieval_batch_size", 128))
        self.retrieval_key_batch_size = int(getattr(args, "retrieval_key_batch_size", 4096))
        self.fusion_weight = float(getattr(args, "fusion_weight", 0.7))
        self.return_mode = getattr(args, "return_pattern_or_value", "value")
        self.enabled = {kind: bool(getattr(args, f"use_{kind}_lib", True)) for kind in PATTERN_TYPES}
        self.dropout = {kind: float(getattr(args, f"{kind}_dropout", 0)) for kind in PATTERN_TYPES}
        self.retrieval_count = {
            kind: int(getattr(args, f"{kind}_retrieval_count", 100000 if kind == "spatiotemporal" else 50000))
            for kind in PATTERN_TYPES
        }
        projection_dim = int(getattr(args, "projection_dim", 64))
        if min(self.feature_dim, projection_dim, self.k_neighbors, self.history_limit,
               self.retrieval_batch_size, self.retrieval_key_batch_size, *self.retrieval_count.values()) < 1:
            raise ValueError("STRAP dimensions and retrieval sizes must be positive")
        if not any(self.enabled.values()):
            raise ValueError("Enable at least one STRAP library, or set use_strap=false")
        if not all(0 <= p <= 1 for p in (*self.dropout.values(), self.history_ratio, self.fusion_weight)):
            raise ValueError("STRAP dropout, history_ratio and fusion_weight must be in [0, 1]")
        if self.return_mode not in ("pattern", "value"):
            raise ValueError(f"Unknown STRAP return mode: {self.return_mode}")

        self.extractor = PatternExtractor(args, self.feature_dim)
        self.projector = RandomProjection(self.feature_dim, projection_dim)
        self.feature_embedding = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.GELU())
        self.pattern_embedding = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.GELU())
        self.topo_feature_embedding = nn.Linear(16, self.feature_dim, bias=False)
        self.banks = nn.ModuleDict({kind: PatternBank(self.feature_dim) for kind in PATTERN_TYPES})
        self.pattern_manager = PatternLibraryManager(args)
        self.current_year = None
        self.source_counts = {}
        self.params = {
            "feature_dim": self.feature_dim, "in_channel": args.gcn["in_channel"],
            "out_channel": args.gcn["out_channel"], "backbone_type": getattr(args, "backbone_type", "stgnn"),
            "enabled": self.enabled, "history_ratio": self.history_ratio, "history_limit": self.history_limit,
            "k_hop": self.extractor.k_hop, "n_clusters": self.extractor.n_clusters,
            "max_neighbors": self.extractor.max_neighbors, "max_cluster_nodes": self.extractor.max_cluster_nodes,
            "time_window": self.extractor.time_window, "overlap": self.extractor.overlap,
            "cross_count": self.extractor.cross_count, "curvature_count": self.extractor.curvature_count,
            "pattern_build_samples": int(getattr(args, "pattern_build_samples", args.batch_size)),
        }
        if self.params["pattern_build_samples"] < 1:
            raise ValueError("pattern_build_samples must be positive")

    def _activate(self, libraries, year, source_counts):
        for kind, bank in self.banks.items():
            bank.assign({name: libraries[kind][name].to(self.projector.weight) for name in ("keys", "values", "topology")})
        self.source_counts = source_counts
        self.current_year = int(year)

    def switch_to_year(self, year):
        year = int(year)
        if self.current_year == year:
            return True
        payload = self.pattern_manager.get_library_for_year(year, self.projector.weight.device)
        if payload is None:
            self.current_year = None
            for bank in self.banks.values():
                bank.assign({name: getattr(bank, name)[:0] for name in ("keys", "values", "topology")})
            return False
        if payload["params"] != self.params:
            raise ValueError(f"STRAP library configuration differs for {year}; rebuild it explicitly")
        self._activate(payload["libraries"], year, payload["source_counts"])
        return True

    def _history(self, kind, year, snapshots, current):
        parts, weights = [], []
        for past_year, payload in snapshots:
            library = payload["libraries"][kind]
            indices = library["history_indices"]
            if len(indices):
                part = {name: library[name][indices] for name in current}
                parts.append(part)
                weights.append(part["keys"].norm(dim=-1).clamp_min(1e-8).reciprocal() * max(0.1, 0.9 ** (year - past_year)))
        if not parts or self.history_ratio == 0:
            return current
        historical = {name: torch.cat([part[name] for part in parts]) for name in current}
        ratio = max(0.1, self.history_ratio * 0.9 ** (year - self.begin_year))
        count = min(len(historical["keys"]), self.history_limit, max(1, int(len(current["keys"]) * ratio)))
        selected = sample_pairs(historical, count, torch.cat(weights))
        return {name: torch.cat((current[name], selected[name])) for name in current}

    @torch.no_grad()
    def extract_patterns(self, data, adj, year, backbone):
        x = data.x.detach().reshape(-1, adj.shape[0], self.params["in_channel"])
        current, source_counts = self.extractor(x, adj, backbone, self.enabled)
        snapshots = []
        for past_year in sorted(self.pattern_manager.metadata, key=int):
            if int(past_year) <= int(year):
                payload = self.pattern_manager.get_library_for_year(int(past_year), x.device)
                if payload["params"] != self.params:
                    raise ValueError(f"STRAP history configuration differs for {past_year}; rebuild the libraries")
                snapshots.append((int(past_year), payload))
        libraries = {}
        for kind in PATTERN_TYPES:
            library = current[kind]
            current_count = len(library["keys"])
            library["origin_year"] = torch.full((current_count,), int(year), device=x.device, dtype=torch.long)
            combined = self._history(kind, int(year), snapshots, library) if self.enabled[kind] else library
            own_year = (combined["origin_year"] == int(year)).nonzero().flatten()
            count = min(self.history_limit, max(1, int(len(own_year) * self.history_ratio))) if len(own_year) and self.history_ratio else 0
            weights = combined["keys"][own_year].norm(dim=-1).clamp_min(1e-8).reciprocal()
            combined["history_indices"] = own_year[torch.multinomial(weights, count, replacement=False)] if count else own_year[:0]
            combined["current_count"] = current_count
            libraries[kind] = combined
        if not sum(len(libraries[kind]["keys"]) for kind in PATTERN_TYPES if self.enabled[kind]):
            raise ValueError("No patterns produced by the enabled STRAP methods for this graph")
        self.pattern_manager.update_library(int(year), libraries, self.params, source_counts)
        self._activate(libraries, year, source_counts)
        return source_counts

    @torch.no_grad()
    def _nearest(self, query, keys, limit):
        distances = query.new_empty(len(query), 0)
        indices = torch.empty(len(query), 0, device=query.device, dtype=torch.long)
        for start in range(0, len(keys), self.retrieval_key_batch_size):
            chunk = keys[start:start + self.retrieval_key_batch_size]
            distance = torch.cdist(query, chunk)
            index = torch.arange(start, start + len(chunk), device=query.device).expand(len(query), -1)
            candidates = torch.cat((distances, distance), 1)
            candidate_indices = torch.cat((indices, index), 1)
            distances, selected = candidates.topk(min(limit, candidates.shape[1]), largest=False, dim=1)
            indices = candidate_indices.gather(1, selected)
        return distances, indices

    def retrieve_patterns(self, x, k=None):
        if self.current_year is None:
            raise RuntimeError("Initialize STRAP from chronological training data before retrieval")
        neighbors = self.k_neighbors if k is None else k
        if neighbors < 1:
            raise ValueError("Retrieval requires at least one neighbor")
        kinds = [kind for kind in PATTERN_TYPES if self.enabled[kind] and len(self.banks[kind].keys)]
        raw_keys = torch.cat([self.banks[kind].keys for kind in kinds])
        values = torch.cat([self.banks[kind].values for kind in kinds])
        topology = torch.cat([self.banks[kind].topology for kind in kinds])
        # The formerly unused topology embedding now participates in key matching
        # and receives gradients through the selected neighbors' distance weights.
        projected_keys = self.projector(raw_keys + self.topo_feature_embedding(topology))
        projected_query = self.projector(x)
        results = []
        for query in projected_query.split(self.retrieval_batch_size):
            scores, indices, masks = [], [], []
            offset = 0
            for kind in kinds:
                count = len(self.banks[kind].keys)
                probability = self.dropout[kind]
                limit = self.retrieval_count[kind]
                if not self.training or probability == 0:
                    limit = min(limit, neighbors)
                distance, index = self._nearest(query, projected_keys[offset:offset + count], limit)
                keep = torch.ones_like(distance, dtype=torch.bool)
                if self.training and probability:
                    keep = (torch.rand_like(distance) >= probability) & (torch.rand(len(query), 1, device=x.device) >= probability)
                scores.append((-distance).masked_fill(~keep, -torch.inf))
                indices.append(index + offset)
                masks.append(keep)
                offset += count
            score = torch.cat(scores, 1)
            selected = score.topk(min(neighbors, score.shape[1]), dim=1).indices
            index = torch.cat(indices, 1).gather(1, selected)
            keep = torch.cat(masks, 1).gather(1, selected)
            # Recompute only selected distances with autograd, avoiding a retained
            # query-by-entire-library graph during training.
            distance = torch.linalg.vector_norm(query[:, None] - projected_keys[index], dim=-1)
            logits = (-distance).masked_fill(~keep, torch.finfo(distance.dtype).min)
            weights = logits.softmax(1) * keep
            weights = weights / weights.sum(1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
            candidates = values if self.return_mode == "value" else raw_keys
            results.append((weights.unsqueeze(-1) * candidates[index]).sum(1))
        return torch.cat(results)

    def forward(self, x):
        retrieved = self.retrieve_patterns(x)
        return self.fusion_weight * self.feature_embedding(x) + (1 - self.fusion_weight) * self.pattern_embedding(retrieved)
