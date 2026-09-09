"""Elastic weight consolidation for the shared training loop."""

import torch
from torch import nn


class EWC(nn.Module):
    def __init__(self, model, adj, ewc_lambda=0, ewc_type="ewc"):
        super().__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.ewc_type = ewc_type
        self.adj = adj

    def register_ewc_params(self, loader, lossfunc, device, with_attention=False):
        parameters = [(name.replace(".", "__"), param)
                      for name, param in self.model.named_parameters() if param.requires_grad]
        fisher = [torch.zeros_like(param) for _, param in parameters]
        for data in loader:
            data.x = data.x.to(device, non_blocking=True)
            data.y = data.y.to(device, non_blocking=True)
            output = self.model(data, self.adj)
            pred = output[0] if with_attention else output
            loss = lossfunc(pred, data.y, reduction="mean")
            gradients = torch.autograd.grad(loss, [param for _, param in parameters])
            for estimate, gradient in zip(fisher, gradients):
                estimate.add_(gradient.detach().square())
        for (name, param), estimate in zip(parameters, fisher):
            self.register_buffer(name + "_estimated_mean", param.detach().clone())
            self.register_buffer(name + "_estimated_fisher", estimate)

    def compute_consolidation_loss(self):
        losses = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            name = name.replace(".", "__")
            mean = getattr(self, name + "_estimated_mean")
            weight = 1e-5 if self.ewc_type == "l2" else getattr(self, name + "_estimated_fisher")
            losses.append((weight * (param - mean).square()).sum())
        return (self.ewc_lambda / 2) * sum(losses)

    def forward(self, data, adj):
        return self.model(data, adj)
