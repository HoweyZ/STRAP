import json
from pathlib import Path

import numpy as np
import torch


def mkdirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json_file(file_path):
    with open(file_path) as file:
        return json.load(file)


def load_best_model(args, *, pattern_year=None):
    if (args.load_first_year and args.year <= args.begin_year + 1) or args.train == 0:
        load_path = Path(args.first_year_model_path)
    else:
        directory = Path(args.path) / str(args.year - 1)
        load_path = min(directory.glob("*.pkl"), key=lambda path: float(path.stem))
    args.logger.info("[*] load from %s", load_path)
    model = args.methods[args.method](args)
    if args.method == "EAC" or (args.method == "Universal" and args.use_eac):
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for size in args.graph_size_list[:args.year - args.begin_year]:
                model.expand_adaptive_params(size)
    # Allocate the destination parameters first, avoiding GPU -> CPU -> GPU loads.
    model = model.to(args.device)
    state = torch.load(load_path, map_location=args.device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    if args.method == "RAP":
        model.set_year(int(load_path.parent.name) if pattern_year is None else pattern_year)
    return model, load_path.stem


def long_term_pattern(args, long_pattern):
    from Bio.Cluster import kcluster

    labels, _, _ = kcluster(long_pattern, nclusters=args.cluster, dist='u')
    return np.eye(args.cluster, dtype=np.float32)[labels]
