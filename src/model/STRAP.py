"""Pattern storage and device-resident STRAP retrieval."""

import pickle
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class PatternLibraryManager:
    """Persist libraries at the disk boundary; active tensors belong to STRAP."""

    def __init__(self, args):
        self.base_dir = Path(args.path) / "pattern_libraries"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, year, pattern_type):
        return self.base_dir / f"{int(year)}_{pattern_type}.pkl"

    def get_library_for_year(self, year, pattern_type="spatiotemporal"):
        path = self._file_path(year, pattern_type)
        if not path.exists():
            return None
        with path.open("rb") as file:
            return pickle.load(file)

    def update_library(self, year, library_data, metadata=None, pattern_type="spatiotemporal"):
        patterns = library_data["patterns"].detach().cpu()
        values = (
            patterns if library_data["values"] is library_data["patterns"]
            else library_data["values"].detach().cpu()
        )
        payload = {"patterns": patterns, "values": values, "metadata": metadata}
        with self._file_path(year, pattern_type).open("wb") as file:
            pickle.dump(payload, file)


class RandomProjection(nn.Module):
    def __init__(self, input_dim, output_dim, seed=42):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(input_dim, output_dim, generator=generator) / output_dim ** 0.5
        self.register_buffer("weight", weight)

    def forward(self, x):
        return x @ self.weight


class STRAP(nn.Module):
    """Exact cosine top-k retrieval with bounded query batches.

    Inputs, keys and values share the module's device and dtype. Pattern files
    hold CPU tensors; loading a year transfers them once. The active library is
    runtime state and follows Module.to() without entering model checkpoints.
    """

    def __init__(self, args):
        super().__init__()
        self.feature_dim = args.gcn["hidden_channel"]
        self.k_neighbors = int(getattr(args, "k_neighbors", 16))
        self.max_patterns = int(getattr(args, "max_patterns", 2048))
        self.retrieval_batch_size = int(getattr(args, "retrieval_batch_size", 1024))
        self.fusion_weight = float(getattr(args, "fusion_weight", 0.7))
        self.return_mode = getattr(args, "return_pattern_or_value", "value")
        if min(self.k_neighbors, self.max_patterns, self.retrieval_batch_size) < 1:
            raise ValueError("Pattern and retrieval sizes must be positive")
        if self.return_mode not in ("pattern", "value"):
            raise ValueError(f"Unknown STRAP return mode: {self.return_mode}")

        self.pattern_manager = PatternLibraryManager(args)
        self.projector = RandomProjection(args.gcn["in_channel"], self.feature_dim)
        self.register_buffer("patterns", torch.empty(0, self.feature_dim), persistent=False)
        self.register_buffer("values", torch.empty(0, self.feature_dim), persistent=False)
        self.current_year = None

    def switch_to_year(self, year):
        year = int(year)
        if self.current_year == year:
            return True
        library = self.pattern_manager.get_library_for_year(year)
        if library is None:
            self.current_year = None
            self.patterns = self.patterns.new_empty(0, self.feature_dim)
            self.values = self.values.new_empty(0, self.feature_dim)
            return False

        patterns = library["patterns"].to(self.projector.weight)
        self.patterns = F.normalize(patterns, dim=-1, eps=1e-8)
        self.values = (
            patterns if library["values"] is library["patterns"]
            else library["values"].to(self.projector.weight)
        )
        self.current_year = year
        return True

    @torch.no_grad()
    def extract_patterns(self, data, adj=None, year=None):
        x = data.x.detach().reshape(-1, data.x.shape[-1])
        # Select before projecting: discarded rows need no projection or normalization.
        if x.shape[0] > self.max_patterns:
            indices = torch.randperm(x.shape[0], device=x.device)[:self.max_patterns]
            x = x[indices]
        self.patterns = F.normalize(self.projector(x), dim=-1, eps=1e-8)
        self.values = self.patterns
        self.current_year = int(year)
        self.pattern_manager.update_library(
            year,
            {"patterns": self.patterns, "values": self.values},
            {"method": "simplified_strap", "num_patterns": self.patterns.shape[0],
             "feature_dim": self.feature_dim},
        )
        return True

    def _retrieve(self, query):
        if self.patterns.shape[0] == 0:
            raise RuntimeError("Initialize the STRAP pattern library before retrieval")
        k = min(self.k_neighbors, self.patterns.shape[0])
        retrieved = []
        for chunk in query.split(self.retrieval_batch_size):
            similarity = F.normalize(chunk, dim=-1, eps=1e-8) @ self.patterns.t()
            scores, indices = similarity.topk(k, dim=1)
            weights = scores.softmax(dim=1).unsqueeze(1)
            retrieved.append(torch.bmm(weights, self.values[indices]).squeeze(1))
        return torch.cat(retrieved, dim=0)

    def forward(self, x):
        retrieved = self._retrieve(x)
        out = self.fusion_weight * x + (1.0 - self.fusion_weight) * retrieved
        if self.return_mode == "pattern":
            return F.normalize(out, dim=-1, eps=1e-8)
        return out
