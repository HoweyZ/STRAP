import math
import pdb
import torch
import torch.nn as nn
import torch.nn.init as init

class BatchGCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True, gcn=True):
        super(BatchGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        if not gcn:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()



    def forward(self, x, adj):
        if x.device != adj.device:
            adj = adj.to(x.device)
    
        input_x = torch.matmul(adj, x)  # [N, N] * [bs, N, in_features] = [bs, N, in_features]
        # x: [bs, N, in_features], adj: [N, N]
            # [N, N] * [bs, N, in_features] = [bs, N, in_features]
        output = self.weight_neigh(input_x)             # [bs, N, in_features] * [in_features, out_features] = [bs, N, out_features]
        if self.weight_self is not None:
            output += self.weight_self(x)               # [bs, N, out_features]
        return output




class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = 3
        self.weight = nn.Parameter(torch.FloatTensor(self.Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)


    def forward(self, x, adj):
        
        x = x.unsqueeze(dim=1)
        
        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')  
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', adj, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', adj, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * adj, x_list[k - 1]) - x_list[k - 2])
        
        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)
        cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        
        x = cheb_graph_conv.squeeze(dim=1)
        
        return x
