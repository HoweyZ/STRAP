"""Shared training and evaluation for prediction and attention models."""

from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from src.dataer.SpatioTemporalDataset import SpatioTemporalDataset
from src.model.ewc import EWC
from utils.common_tools import load_best_model
from utils.metric import ForecastMetrics, log_metrics, masked_mae


def _loader(inputs, split, args, nodes=None):
    if nodes is None:
        dataset = SpatioTemporalDataset(inputs, split)
    else:
        dataset = SpatioTemporalDataset(
            "", "", x=inputs[split + "_x"][:, :, nodes],
            y=inputs[split + "_y"][:, :, nodes], mode="subgraph",
        )
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=split == "train",
        pin_memory=torch.device(args.device).type == "cuda",
        num_workers=getattr(args, "num_workers", 0),
    )


def _move_batch(data, device, non_blocking=True):
    # Batch/ptr metadata is unused: each sample has the same graph and node count.
    data.x = data.x.to(device, non_blocking=non_blocking)
    data.y = data.y.to(device, non_blocking=non_blocking)
    return data


def _predict(model, data, adj, with_attention):
    output = model(data, adj)
    return output if with_attention else (output, None)


def _targets(pred, target, n_nodes, mapping):
    if mapping is not None:
        pred = pred.reshape(-1, n_nodes, pred.shape[-1])[:, mapping]
        target = target.reshape(-1, n_nodes, target.shape[-1])[:, mapping]
    return pred, target


def train(inputs, args, with_attention=False):
    path = Path(args.path) / str(args.year)
    path.mkdir(parents=True, exist_ok=True)
    lossfunc = {"mse": F.mse_loss, "huber": F.smooth_l1_loss}[args.loss]
    incremental = args.strategy == "incremental" and args.year > args.begin_year
    mapping = None
    nodes = None
    if incremental:
        nodes = args.subgraph.numpy()
        edges = args.subgraph_edge_index.to(args.device)
        adj = torch.zeros(len(nodes), len(nodes), device=args.device)
        adj[edges[0], edges[1]] = 1
        adj[edges[1], edges[0]] = 1
        args.sub_adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)
        mapping = args.mapping.to(args.device)
    else:
        args.sub_adj = args.adj
    if with_attention:
        args.past_adj = args.sub_adj

    train_loader = _loader(inputs, "train", args, nodes)
    val_loader = _loader(inputs, "val", args, nodes)
    test_loader = _loader(inputs, "test", args)
    args.logger.info("[*] Year %s Dataset load!", args.year)

    continuing = args.init and args.year > args.begin_year
    if continuing:
        gnn_model, _ = load_best_model(args, pattern_year=args.year)
    else:
        gnn_model = args.methods[args.method](args).to(args.device)
    adaptive = args.method == "EAC" or (args.method == "Universal" and args.use_eac)
    if adaptive:
        if continuing:
            for name, param in gnn_model.named_parameters():
                if any(part in name for part in ("gcn1", "tcn1", "gcn2", "fc")):
                    param.requires_grad_(False)
        gnn_model.expand_adaptive_params(args.graph_size)

    model = gnn_model
    if continuing and args.ewc:
        model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)
        model.register_ewc_params(
            _loader(inputs, "train", args), lossfunc, args.device,
            with_attention=with_attention,
        )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    attention_labels = None
    if with_attention and args.year == args.begin_year:
        attention_labels = torch.as_tensor(args.attention, device=args.device).argmax(dim=1)

    best_loss = float("inf")
    counter = 0
    use_time = []
    args.logger.info("[*] Year %s Training start", args.year)
    for epoch in range(args.epoch):
        model.train()
        start = perf_counter()
        training_loss = torch.zeros((), device=args.device)
        for data in train_loader:
            data = _move_batch(data, args.device)
            optimizer.zero_grad(set_to_none=True)
            pred, attention = _predict(model, data, args.sub_adj, with_attention)
            pred, target = _targets(pred, data.y, args.sub_adj.shape[0], mapping)
            loss = lossfunc(pred, target)
            if attention_labels is not None:
                labels = attention_labels.repeat(data.x.shape[0] // args.sub_adj.shape[0])
                loss = loss + 0.1 * F.cross_entropy(attention, labels)
            if continuing and args.ewc:
                loss = loss + model.compute_consolidation_loss()
            training_loss.add_(loss.detach())
            loss.backward()
            optimizer.step()
        # Synchronize once per epoch, rather than float(loss) at every batch.
        training_loss = (training_loss / len(train_loader)).item()
        use_time.append(perf_counter() - start)

        model.eval()
        validation_loss = torch.zeros((), device=args.device)
        with torch.no_grad():
            for data in val_loader:
                data = _move_batch(data, args.device)
                pred, _ = _predict(model, data, args.sub_adj, with_attention)
                pred, target = _targets(pred, data.y, args.sub_adj.shape[0], mapping)
                validation_loss.add_(masked_mae(target, pred))
        validation_loss = (validation_loss / len(val_loader)).item()
        args.logger.info(
            "epoch:%d, training loss:%.4f validation loss:%.4f",
            epoch, training_loss, validation_loss,
        )
        if validation_loss <= best_loss:
            counter = 0
            best_loss = round(validation_loss, 4)
            best_path = path / f"{best_loss}.pkl"
            torch.save({"model_state_dict": gnn_model.state_dict()}, best_path)
        else:
            counter += 1
            if counter > 5:
                break

    gnn_model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True)["model_state_dict"])
    test_model(gnn_model, args, test_loader, True, with_attention=with_attention)
    total_time = sum(use_time)
    args.result[args.year] = {
        "total_time": total_time, "average_time": total_time / len(use_time),
        "epoch_num": len(use_time),
    }
    args.logger.info("Finished optimization, total time:%.2f s, best model:%s", total_time, best_path)


@torch.no_grad()
def test_model(model, args, testset, pin_memory, with_attention=False):
    model.eval()
    metrics = ForecastMetrics(args.y_len, args.device)
    loss = torch.zeros((), device=args.device)
    for data in testset:
        data = _move_batch(data, args.device, non_blocking=pin_memory)
        pred, _ = _predict(model, data, args.adj, with_attention)
        loss.add_(F.mse_loss(pred, data.y))
        # Samples have equal node counts; reshape avoids PyG's host size inference.
        metrics.update(
            data.y.reshape(-1, args.adj.shape[0], args.y_len),
            pred.reshape(-1, args.adj.shape[0], args.y_len),
        )
    args.logger.info("[*] loss:%.4f", (loss / len(testset)).item())
    log_metrics(metrics, args)
