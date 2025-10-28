import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn_conv import BatchGCNConv, ChebGraphConv
from scipy.sparse.linalg import eigs
from model.STRAP import STRAP
import os
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from torch.nn.utils.rnn import pad_sequence
import time
import math


# -----------------------------------------------
# backbone
# -----------------------------------------------

class STGNN_Backbone(nn.Module):
    def __init__(self, args):
        super(STGNN_Backbone, self).__init__()
       
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        
        self.tcn = nn.Conv1d(
            in_channels=args.tcn["in_channel"], 
            out_channels=args.tcn["out_channel"], 
            kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"], 
            padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2)
        )
        
    def forward(self, x, adj):
    
        x = torch.nn.functional.relu(self.gcn1(x, adj))  # [bs, N, hidden]
        
        B, N, H = x.shape
        
     
        x = x.reshape(B*N, 1, H)  # [bs*N, 1, hidden]
        x = self.tcn(x)  # [bs*N, 1, hidden]
        
    
        x = x.reshape(B, N, H)  # [bs, N, hidden]
        x = self.gcn2(x, adj)  # [bs, N, out]
        
        return x

class DCRNN_Backbone(nn.Module):
 
    def __init__(self, args):
        super(DCRNN_Backbone, self).__init__()
    
        self.diffusion_conv_forward = BatchGCNConv(
            args.gcn["in_channel"], 
            args.gcn["hidden_channel"]//2, 
            bias=True, 
            gcn=False
        )
        self.diffusion_conv_backward = BatchGCNConv(
            args.gcn["in_channel"], 
            args.gcn["hidden_channel"]//2, 
            bias=True, 
            gcn=False
        )
        
     
        self.gru_cell = nn.GRUCell(
            args.gcn["hidden_channel"], 
            args.gcn["hidden_channel"]
        )
        
      
        self.diffusion_conv_out = BatchGCNConv(
            args.gcn["hidden_channel"], 
            args.gcn["out_channel"], 
            bias=True, 
            gcn=False
        )
        
    def forward(self, x, adj):
        B, N, F = x.shape
        
   
        backward_adj = adj.transpose(0, 1)
        

        forward_diff = torch.nn.functional.relu(self.diffusion_conv_forward(x, adj))  # [bs, N, hidden/2]
        backward_diff = torch.nn.functional.relu(self.diffusion_conv_backward(x, backward_adj))  # [bs, N, hidden/2]
        
    
        diff_features = torch.cat([forward_diff, backward_diff], dim=-1)  # [bs, N, hidden]
        diff_features_flat = diff_features.reshape(B*N, -1)  # [bs*N, hidden]
        
      
        h = torch.zeros_like(diff_features_flat)  # [bs*N, hidden]
        
      
        h = self.gru_cell(diff_features_flat, h)  # [bs*N, hidden]
        
       
        h = h.reshape(B, N, -1)  # [bs, N, hidden]
        
       
        x = self.diffusion_conv_out(h, adj)  # [bs, N, out]
        
        return x


class ASTGNN_Backbone(nn.Module):

    def __init__(self, args):
        super(ASTGNN_Backbone, self).__init__()
      
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        
      
        self.attention_layer = nn.Sequential(
            nn.Linear(args.gcn["in_channel"], args.gcn["hidden_channel"]),
            nn.ReLU(),
            nn.Linear(args.gcn["hidden_channel"], 1)
        )
        
       
        self.tcn = nn.Conv1d(
            in_channels=args.tcn["in_channel"], 
            out_channels=args.tcn["out_channel"], 
            kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"], 
            padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2)
        )
        
    def _compute_adaptive_adj(self, x, adj):
     
        B, N, F = x.shape
        
      
        x_flat = x.reshape(B*N, F)  # [bs*N, F]
        
       
        attention_scores = self.attention_layer(x_flat).squeeze(-1)  # [bs*N]
        attention_scores = attention_scores.reshape(B, N)  # [bs, N]
        
       
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)  # [bs, N]
        
        
        weighted_adj = adj.unsqueeze(0) * attention_weights.unsqueeze(-1)  # [bs, N, N]
        
       
        adaptive_adj = weighted_adj.mean(dim=0)  # [N, N]
        
        return adaptive_adj
        
    def forward(self, x, adj):
        B, N, F = x.shape
        
       
        adaptive_adj = self._compute_adaptive_adj(x, adj)
        
      
        x = torch.nn.functional.relu(self.gcn1(x, adaptive_adj))  # [bs, N, hidden]
        
    
        x = x.reshape(B*N, 1, -1)  # [bs*N, 1, hidden]
        x = self.tcn(x)  # [bs*N, 1, hidden]
        
        
        x = x.reshape(B, N, -1)  # [bs, N, hidden]
        x = self.gcn2(x, adaptive_adj)  # [bs, N, out]
        
        return x

class TGCN_Backbone(nn.Module):

    def __init__(self, args):
        super(TGCN_Backbone, self).__init__()
        self.args = args
        
      
        self.gcn = BatchGCNConv(
            args.gcn["in_channel"], 
            args.gcn["hidden_channel"], 
            bias=True, 
            gcn=False
        )
        
  
        self.input_size = args.gcn["hidden_channel"]
        self.hidden_size = args.gcn["hidden_channel"]
        
 
        self.weight_xz = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hz = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.weight_xr = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hr = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.weight_xh = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hh = nn.Linear(self.hidden_size, self.hidden_size)
        
 
        self.output_layer = nn.Linear(self.hidden_size, args.gcn["out_channel"])
        
    
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, adj):
   
        batch_size, num_nodes, input_dim = x.shape
        

        h = torch.zeros(batch_size, num_nodes, self.hidden_size, device=x.device)
        
   
        x_gcn = self.gcn(x, adj)  # [batch_size, num_nodes, hidden_size]
        
 
        z = self.sigmoid(self.weight_xz(x_gcn) + self.weight_hz(h))  
        r = self.sigmoid(self.weight_xr(x_gcn) + self.weight_hr(h))  
        h_tilde = self.activation(self.weight_xh(x_gcn) + self.weight_hh(r * h))  
        h = (1 - z) * h + z * h_tilde  
        
  
        output = self.output_layer(h)  # [batch_size, num_nodes, output_dim]
        
        return output


# -----------------------------------------------
# backbone_base
# -----------------------------------------------

class STGNN_Model(nn.Module):
 
    def __init__(self, args):
        super(STGNN_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout if hasattr(args, 'dropout') else 0.1
        
  
        self.backbone = STGNN_Backbone(args)
        
 
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if hasattr(self.args, 'logger'):
            self.args.logger.info(f"Total Parameters: {total_params}")
            self.args.logger.info(f"Trainable Parameters: {trainable_params}")
        else:
            print(f"Total Parameters: {total_params}")
            print(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
      
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
        
        feature_map = self.backbone(x, adj)
        
    
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
   
        x = feature_map + data.x
        
    
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def feature(self, data, adj):
        """提取特征但不进行最终预测"""
        N = adj.shape[0]
        

        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        
 
        feature_map = self.backbone(x, adj)
        
        
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
    
        x = feature_map + data.x
        return x


class DCRNN_Model(nn.Module):
   
    def __init__(self, args):
        super(DCRNN_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout if hasattr(args, 'dropout') else 0.1
        
       
        self.backbone = DCRNN_Backbone(args)
        
      
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if hasattr(self.args, 'logger'):
            self.args.logger.info(f"Total Parameters: {total_params}")
            self.args.logger.info(f"Trainable Parameters: {trainable_params}")
        else:
            print(f"Total Parameters: {total_params}")
            print(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
      
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
    
        feature_map = self.backbone(x, adj)
        
  
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
   
        x = feature_map + data.x
     
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def feature(self, data, adj):
     
        N = adj.shape[0]
        

        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        
   
        feature_map = self.backbone(x, adj)
        
       
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
        
        x = feature_map + data.x
        return x


class ASTGNN_Model(nn.Module):
  
    def __init__(self, args):
        super(ASTGNN_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout if hasattr(args, 'dropout') else 0.1
        
      
        self.backbone = ASTGNN_Backbone(args)
        

        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if hasattr(self.args, 'logger'):
            self.args.logger.info(f"Total Parameters: {total_params}")
            self.args.logger.info(f"Trainable Parameters: {trainable_params}")
        else:
            print(f"Total Parameters: {total_params}")
            print(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
  
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
   
        feature_map = self.backbone(x, adj)
        

        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        

        x = feature_map + data.x
        

        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def feature(self, data, adj):
    
        N = adj.shape[0]
        
     
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        
     
        feature_map = self.backbone(x, adj)
        
      
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
       
        x = feature_map + data.x
        return x
    

class TGCN_Model(nn.Module):

    def __init__(self, args):
        super(TGCN_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout if hasattr(args, 'dropout') else 0.1
        
        self.gcn = BatchGCNConv(
            args.gcn["in_channel"], 
            args.gcn["hidden_channel"], 
            bias=True, 
            gcn=False
        )
        
        self.input_size = args.gcn["hidden_channel"]
        self.hidden_size = args.gcn["hidden_channel"]

        self.weight_xz = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hz = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.weight_xr = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hr = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.weight_xh = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hh = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.output_layer = nn.Linear(self.hidden_size, args.y_len)
        
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if hasattr(self.args, 'logger'):
            self.args.logger.info(f"Total Parameters: {total_params}")
            self.args.logger.info(f"Trainable Parameters: {trainable_params}")
        else:
            print(f"Total Parameters: {total_params}")
            print(f"Trainable Parameters: {trainable_params}")
        
    def forward(self, data, adj):
       
        N = adj.shape[0]
        
      
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        batch_size, num_nodes, input_dim = x.shape
        
        
        h = torch.zeros(batch_size, num_nodes, self.hidden_size, device=x.device)
             
        x_gcn = self.gcn(x, adj)  # [batch_size, num_nodes, hidden_size]
        
      
        z = self.sigmoid(self.weight_xz(x_gcn) + self.weight_hz(h)) 
        r = self.sigmoid(self.weight_xr(x_gcn) + self.weight_hr(h))  
        h_tilde = self.activation(self.weight_xh(x_gcn) + self.weight_hh(r * h))  
        h = (1 - z) * h + z * h_tilde  
        
      
        feature_map = h.reshape(-1, self.hidden_size)
        
     
        output = self.output_layer(feature_map)
        output = F.dropout(output, p=self.dropout, training=self.training)
        
        return output
    
    def feature(self, data, adj):
       
        N = adj.shape[0]
        
       
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        batch_size, num_nodes, input_dim = x.shape
        
      
        h = torch.zeros(batch_size, num_nodes, self.hidden_size, device=x.device)
        
      
        x_gcn = self.gcn(x, adj)  # [batch_size, num_nodes, hidden_size]
        
      
        z = self.sigmoid(self.weight_xz(x_gcn) + self.weight_hz(h))  
        r = self.sigmoid(self.weight_xr(x_gcn) + self.weight_hr(h))  
        h_tilde = self.activation(self.weight_xh(x_gcn) + self.weight_hh(r * h))  
        h = (1 - z) * h + z * h_tilde  
        
        
        return h.reshape(-1, self.hidden_size)


   
# -----------------------------------------------
# LoRA layer
# -----------------------------------------------

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=10):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_a = nn.init.xavier_uniform_(nn.Parameter(torch.empty(in_dim, r)))
        self.lora_b = nn.Parameter(torch.zeros(r, out_dim))
        self.scaling = 1 / (r * in_dim)

    def forward(self, x):
        return x + self.scaling * torch.matmul(torch.matmul(x, self.lora_a.to(x.device)), self.lora_b.to(x.device))
    
# -----------------------------------------------
# Adapter layer
# -----------------------------------------------
class AdapterLayer(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim, dropout_rate=0.1):
        super(AdapterLayer, self).__init__()
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
       
        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        
        residual = x
        
       
        x = self.layer_norm(x)
        
        
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        
       
        out = residual + x
        
        return out
    
# -----------------------------------------------
# baseline
# -----------------------------------------------

class GraphPro_Model(nn.Module):
    def __init__(self, args):
        super(GraphPro_Model, self).__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        
        
        self.logger = getattr(args, "logger", None)
        
       
        backbone_type = getattr(args, "backbone_type", "stgnn")
        if backbone_type == "dcrnn":
            self.backbone = DCRNN_Backbone(args)
        elif backbone_type == "astgnn":
            self.backbone = ASTGNN_Backbone(args)
        elif backbone_type == "tgcn":
            self.backbone = TGCN_Backbone(args)
        else: 
            self.backbone = STGNN_Backbone(args)
        
       
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
       
        self.current_phase = "pretrain"  
        self.phase_steps = 0  
        self.total_steps = 0 
        
     
        total_epochs = getattr(args, "total_epochs", 100)  
        pretrain_ratio = getattr(args, "pretrain_ratio", 0.5)  
        for_tune_ratio = getattr(args, "for_tune_ratio", 0.2)  
        finetune_ratio = getattr(args, "finetune_ratio", 0.3)  
        
      
        total_ratio = pretrain_ratio + for_tune_ratio + finetune_ratio
        if abs(total_ratio - 1.0) > 1e-5:
            
            pretrain_ratio /= total_ratio
            for_tune_ratio /= total_ratio
            finetune_ratio /= total_ratio
        
       
        self.phase_schedule = {
            "pretrain": int(total_epochs * pretrain_ratio),
            "for_tune": int(total_epochs * for_tune_ratio),
            "finetune": int(total_epochs * finetune_ratio)
        }
        
      
        actual_total = sum(self.phase_schedule.values())
        if actual_total < total_epochs:
          
            self.phase_schedule["finetune"] += (total_epochs - actual_total)
        
       
        self.freeze_in_finetune = getattr(args, "freeze_in_finetune", True)
        
        
        self.lr_schedule = {
            "pretrain": getattr(args, "pretrain_lr", 0.03),
            "for_tune": getattr(args, "for_tune_lr", 0.015),
            "finetune": getattr(args, "finetune_lr", 0.003)
        }
        
       
        self.gating_weight = None
        self.gating_bias = None
        self.emb_dropout = None
        
       
        input_dim = args.gcn["in_channel"]
        self.prompt_pool_dense = nn.Linear(input_dim, input_dim)
        
       
        self._setup_current_phase()
        
       
        self.use_time_encoding = getattr(args, "use_time_encoding", True)
        if self.use_time_encoding:
            self.max_time_step = getattr(args, "max_time_step", None)
        
     
        self.edge_dropout = getattr(args, "edge_dropout", 0.0)
        
       
        if self.logger:
            self.logger.info(f"GraphPro initialized with backbone {backbone_type}")
            self.logger.info(f"Initial phase: {self.current_phase}")
            self.logger.info(f"Phase schedule: {self.phase_schedule}")
            self.logger.info(f"Time encoding: {self.use_time_encoding}")
        else:
            print(f"GraphPro initialized with backbone {backbone_type}")
            print(f"Initial phase: {self.current_phase}")
            print(f"Phase schedule: {self.phase_schedule}")
            print(f"Time encoding: {self.use_time_encoding}")
    
    def _setup_current_phase(self):
      
        if self.current_phase == "finetune":
          
            if self.gating_weight is None:
                input_dim = self.args.gcn["in_channel"]
                self.gating_weight = nn.Parameter(
                    nn.init.xavier_uniform_(torch.empty(input_dim, input_dim))
                )
                self.gating_bias = nn.Parameter(
                    nn.init.zeros_(torch.empty(1, input_dim))
                )
                self.emb_dropout = nn.Dropout(getattr(self.args, "emb_dropout", 0.1))
            
           
            self.emb_gate = lambda x: self.emb_dropout(
                torch.mul(x, torch.sigmoid(torch.matmul(x, self.gating_weight) + self.gating_bias))
            )
            
         
            if self.freeze_in_finetune:
                self.freeze_backbone()
                
        elif self.current_phase == "for_tune":
           
            self.emb_gate = self.random_gate
            
           
            self.unfreeze_all()
            
        else: 
           
            self.emb_gate = lambda x: x
            
           
            self.unfreeze_all()
        
        
        self.phase_steps = 0
        
        if self.logger:
            self.logger.info(f"Phase set to: {self.current_phase}")
            self.logger.info(f"Phase will last for {self.phase_schedule[self.current_phase]} epochs")
        else:
            print(f"Phase set to: {self.current_phase}")
            print(f"Phase will last for {self.phase_schedule[self.current_phase]} epochs")
    
    def step_epoch(self, optimizer=None, force_next_phase=False):
    
        self.phase_steps += 1
        self.total_steps += 1
        phase_changed = False
        
      
        if force_next_phase or (self.current_phase == "pretrain" and self.phase_steps >= self.phase_schedule["pretrain"]):
            self.current_phase = "for_tune"
            self._setup_current_phase()
            phase_changed = True
            
        elif force_next_phase or (self.current_phase == "for_tune" and self.phase_steps >= self.phase_schedule["for_tune"]):
            self.current_phase = "finetune"
            self._setup_current_phase()
            phase_changed = True
        
     
        if phase_changed and optimizer is not None:
            new_lr = self.lr_schedule[self.current_phase]
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            if self.logger:
                self.logger.info(f"Learning rate updated to: {new_lr}")
            else:
                print(f"Learning rate updated to: {new_lr}")
        
        return phase_changed
    
    def force_next_phase(self, optimizer=None):
       
        return self.step_epoch(optimizer, force_next_phase=True)
    
    def get_training_progress(self):
        
        return {
            "current_phase": self.current_phase,
            "phase_steps": self.phase_steps,
            "total_steps": self.total_steps,
            "phase_schedule": self.phase_schedule,
            "remaining_in_phase": max(0, self.phase_schedule[self.current_phase] - self.phase_steps)
        }
    
    def load_state_dict(self, state_dict, strict=True):
      
        if 'prompt_pool_dense.weight' in state_dict and 'gating_weight' not in state_dict:
            
            
            if self.current_phase == "finetune" and hasattr(self, 'gating_weight') and hasattr(self, 'gating_bias'):
               
                new_state_dict = state_dict.copy()
                
             
                if 'gating_weight' not in new_state_dict:
                    new_state_dict['gating_weight'] = state_dict['prompt_pool_dense.weight'].clone()
                if 'gating_bias' not in new_state_dict:
                    new_state_dict['gating_bias'] = state_dict['prompt_pool_dense.bias'].clone().reshape(1, -1)
                
               
                return super(GraphPro_Model, self).load_state_dict(new_state_dict, strict=False)
        
        return super(GraphPro_Model, self).load_state_dict(state_dict, strict=strict)
    
    def count_parameters(self):
      
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
      
        gating_params = 0
        if self.current_phase == "finetune" and self.gating_weight is not None:
            gating_params = sum(p.numel() for p in [self.gating_weight, self.gating_bias])
        
        if self.logger:
            self.logger.info(f"Total Parameters: {total_params}")
            self.logger.info(f"Trainable Parameters: {trainable_params}")
            if self.current_phase == "finetune":
                self.logger.info(f"Gating Parameters: {gating_params}")
        else:
            print(f"Total Parameters: {total_params}")
            print(f"Trainable Parameters: {trainable_params}")
            if self.current_phase == "finetune":
                print(f"Gating Parameters: {gating_params}")
    
    def random_gate(self, x):

        device = x.device
        input_dim = x.shape[-1]
        
        
        gating_weight = F.normalize(
            torch.randn((input_dim, input_dim), device=device)
        )
        gating_bias = F.normalize(
            torch.randn((1, input_dim), device=device)
        )
        
        
        gate = torch.sigmoid(torch.matmul(x, gating_weight) + gating_bias)
        
        return torch.mul(x, gate)
    
    def _relative_time_encoding(self, adj, edge_times=None, max_step=None):
       
        if edge_times is None or not self.use_time_encoding:
            return adj
        
       
        N = adj.shape[0]
        edges = (adj > 0).nonzero(as_tuple=False)  # [E, 2]
        
        times = edge_times.float()
        
        if max_step is None:
            max_step = times.max()
        
        min_time = times.min()
        if max_step > min_time:
            normalized_times = (times - min_time) / (max_step - min_time)
        else:
            normalized_times = torch.zeros_like(times)
        
        dst_nodes = edges[:, 1]
        time_weights = torch.zeros_like(normalized_times)
        
        for i in range(N):
            mask = (dst_nodes == i)
            if mask.sum() > 0:
                node_times = normalized_times[mask]
               
                node_weights = F.softmax(node_times, dim=0)
                time_weights[mask] = node_weights
        
     
        time_adj = torch.zeros_like(adj)
        time_adj[edges[:, 0], edges[:, 1]] = time_weights
        
       
        combined_adj = adj * 0.5 + time_adj * 0.5
        
        return combined_adj
    
    def _edge_dropout(self, adj, dropout_rate):
       
        if dropout_rate <= 0:
            return adj

        mask = torch.rand_like(adj) > dropout_rate

        dropped_adj = adj * mask.float()
        
        return dropped_adj
    
    def set_phase(self, phase):
       
        self.current_phase = phase
        self._setup_current_phase()
    
    def freeze_backbone(self):
      
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.fc.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
  
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, data, adj, edge_times=None):
        N = adj.shape[0]
        
      
        if self.training and self.edge_dropout > 0:
            adj = self._edge_dropout(adj, self.edge_dropout)
        
       
        if self.use_time_encoding and edge_times is not None:
            adj = self._relative_time_encoding(adj, edge_times, self.max_time_step)
        
      
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
        
        B, N, C = x.shape
        x_flat = x.reshape(-1, C)
        x_gated = self.emb_gate(x_flat)
        x = x_gated.reshape(B, N, C)
        
       
        feature_map = self.backbone(x, adj)
   
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
    
        x = feature_map + data.x
        
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

    def feature(self, data, adj, edge_times=None):
       
        N = adj.shape[0]
        
        if self.use_time_encoding and edge_times is not None:
            adj = self._relative_time_encoding(adj, edge_times, self.max_time_step)
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        
        B, N, C = x.shape
        x_flat = x.reshape(-1, C)
        x_gated = self.emb_gate(x_flat)
        x = x_gated.reshape(B, N, C)
        
        feature_map = self.backbone(x, adj)
        
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
        x = feature_map + data.x
        return x


class PECPM_Model(nn.Module):
    def __init__(self, args):
        super(PECPM_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        
        
        if not hasattr(args, "attention_weight"):
            self.top_k = 5  
        elif isinstance(args.attention_weight, dict):
            self.top_k = 5  
        else:
            self.top_k = args.attention_weight
        
       
        self.historical_patterns = None
        
        backbone_type = getattr(args, "backbone_type", "stgnn")
        if backbone_type == "dcrnn":
            self.backbone = DCRNN_Backbone(args)
        elif backbone_type == "astgnn":
            self.backbone = ASTGNN_Backbone(args)
        elif backbone_type == "tgcn":
            self.backbone = TGCN_Backbone(args)
        else:  
            self.backbone = STGNN_Backbone(args)
        
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        self.logger = getattr(args, "logger", None)
        if self.logger:
            self.logger.info(f"PECPM initialized with backbone {backbone_type}")

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def pattern_matching(self, current_features):
    
        if self.historical_patterns is None:
            self.historical_patterns = current_features.detach().cpu()
            return torch.ones(current_features.size(0), 1, device=current_features.device)
        
        if hasattr(self.args, "attention_weight") and isinstance(self.args.attention_weight, dict):
            year_offset = str(self.args.year - self.args.begin_year)
            self.top_k = self.args.attention_weight.get(year_offset, 5)
        
        current_cpu = current_features.detach().cpu()
        
        current_norm = torch.nn.functional.normalize(current_cpu, p=2, dim=1)
        history_norm = torch.nn.functional.normalize(self.historical_patterns, p=2, dim=1)
        
        similarity = torch.mm(current_norm, history_norm.t())
        
    
        topk_values, _ = similarity.topk(min(self.top_k, similarity.size(1)), dim=1)
        
      
        pattern_scores = topk_values.mean(dim=1).unsqueeze(1).to(current_features.device)
        
        
        self.historical_patterns = torch.cat([self.historical_patterns, current_cpu], dim=0)
        
        max_patterns = 1000 
        if self.historical_patterns.size(0) > max_patterns:
            self.historical_patterns = self.historical_patterns[-max_patterns:]
        
        return pattern_scores

    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feature_map = self.backbone(x, adj)
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
        pattern_scores = self.pattern_matching(feature_map)
        
        enhanced_features = feature_map * pattern_scores
        
        x = enhanced_features + data.x
        
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feature_map = self.backbone(x, adj)
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
    
        pattern_scores = self.pattern_matching(feature_map)
        
        enhanced_features = feature_map * pattern_scores
        
        x = enhanced_features + data.x
        return x
    
class STAdapter_Model(nn.Module):

    def __init__(self, args):
        super(STAdapter_Model, self).__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        
        backbone_type = getattr(args, "backbone_type", "stgnn")
        if backbone_type == "dcrnn":
            self.backbone = DCRNN_Backbone(args)
        elif backbone_type == "astgnn":
            self.backbone = ASTGNN_Backbone(args)
        elif backbone_type == "tgcn":
            self.backbone = TGCN_Backbone(args)
        else: 
            self.backbone = STGNN_Backbone(args)
        
       
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
    
        self.bottleneck_ratio = getattr(args, "bottleneck_ratio", 8)  
        self.input_adapter_dim = args.gcn["in_channel"] // self.bottleneck_ratio
        self.hidden_adapter_dim = args.gcn["hidden_channel"] // self.bottleneck_ratio
        self.output_adapter_dim = args.gcn["out_channel"] // self.bottleneck_ratio
        
        
        self.input_adapters = nn.ModuleList()   
        self.hidden_adapters = nn.ModuleList()  
        self.output_adapters = nn.ModuleList()  
        
       
        self.add_adapter_group()
        
        self.logger = getattr(args, "logger", None)
        if self.logger:
            self.logger.info(f"ST-Adapter initialized with backbone {backbone_type}")
            self.logger.info(f"Adapter bottleneck ratio: 1/{self.bottleneck_ratio}")

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        input_adapter_params = sum(p.numel() for adapter in self.input_adapters for p in adapter.parameters())
        hidden_adapter_params = sum(p.numel() for adapter in self.hidden_adapters for p in adapter.parameters())
        output_adapter_params = sum(p.numel() for adapter in self.output_adapters for p in adapter.parameters())
        total_adapter_params = input_adapter_params + hidden_adapter_params + output_adapter_params

        if self.logger:
            self.logger.info(f"Total Parameters: {total_params}")
            self.logger.info(f"Trainable Parameters: {trainable_params}")
            self.logger.info(f"Total Adapter Parameters: {total_adapter_params}")
            self.logger.info(f"- Input Adapter Params: {input_adapter_params}")
            self.logger.info(f"- Hidden Adapter Params: {hidden_adapter_params}")
            self.logger.info(f"- Output Adapter Params: {output_adapter_params}")
        else:
            print(f"Total Parameters: {total_params}")
            print(f"Trainable Parameters: {trainable_params}")
            print(f"Total Adapter Parameters: {total_adapter_params}")
    
    def add_adapter_group(self):
       
        input_adapter = AdapterLayer(
            hidden_dim=self.args.gcn["in_channel"],
            bottleneck_dim=self.input_adapter_dim,
            dropout_rate=self.dropout
        )
        
        hidden_adapter = AdapterLayer(
            hidden_dim=self.args.gcn["hidden_channel"],
            bottleneck_dim=self.hidden_adapter_dim,
            dropout_rate=self.dropout
        )
    
        output_adapter = AdapterLayer(
            hidden_dim=self.args.gcn["out_channel"],
            bottleneck_dim=self.output_adapter_dim,
            dropout_rate=self.dropout
        )
        
        self.input_adapters.append(input_adapter)
        self.hidden_adapters.append(hidden_adapter)
        self.output_adapters.append(output_adapter)
        
        self._freeze_previous_adapters()
    
    def _freeze_previous_adapters(self):

        if len(self.input_adapters) > 1:
            for i in range(len(self.input_adapters) - 1):
                for param in self.input_adapters[i].parameters():
                    param.requires_grad = False
                for param in self.hidden_adapters[i].parameters():
                    param.requires_grad = False
                for param in self.output_adapters[i].parameters():
                    param.requires_grad = False
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.fc.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def _apply_adapters(self, x, adapters):
        for adapter in adapters:
            x = adapter(x)
        return x

    def forward(self, data, adj):

        N = adj.shape[0]
        
        x_original = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        bs = x_original.shape[0]
        
        x_flat = x_original.reshape(-1, self.args.gcn["in_channel"])
        x_flat = self._apply_adapters(x_flat, self.input_adapters)
        x = x_flat.reshape(bs, N, -1)
        
        hidden_features = None
        
        def hook_fn(module, input, output):
            nonlocal hidden_features
            if isinstance(output, tuple):
                output = output[0]  
            
            if len(output.shape) == 3:  # [bs, N, hidden]
                hidden_features = output
            else:  
                try:
                    hidden_features = output.reshape(bs, N, -1)
                except:

                    hidden_features = output
        

        hook_module = None
        for name, module in self.backbone.named_modules():
            if "gcn" in name.lower() or isinstance(module, BatchGCNConv):
                hook_module = module
                break
        
        hook_handle = None
        if hook_module:
            hook_handle = hook_module.register_forward_hook(hook_fn)
        
        output_features = self.backbone(x, adj)
        
        if hook_handle:
            hook_handle.remove()
        
        if hidden_features is not None:
            hidden_flat = hidden_features.reshape(-1, hidden_features.shape[-1])
            if hidden_flat.shape[-1] == self.args.gcn["hidden_channel"]:
                hidden_flat = self._apply_adapters(hidden_flat, self.hidden_adapters)
        
        output_features = output_features.reshape(-1, self.args.gcn["out_channel"])
        
        output_features = self._apply_adapters(output_features, self.output_adapters)
        
        x = output_features + data.x
        
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        
        x_original = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        bs = x_original.shape[0]
        
        x_flat = x_original.reshape(-1, self.args.gcn["in_channel"])
        x_flat = self._apply_adapters(x_flat, self.input_adapters)
        x = x_flat.reshape(bs, N, -1)
        
        hidden_features = None
        
        def hook_fn(module, input, output):
            nonlocal hidden_features
            if isinstance(output, tuple):
                output = output[0]
            
            if len(output.shape) == 3:
                hidden_features = output
            else:
                try:
                    hidden_features = output.reshape(bs, N, -1)
                except:
                    hidden_features = output
        
        hook_module = None
        for name, module in self.backbone.named_modules():
            if "gcn" in name.lower() or isinstance(module, BatchGCNConv):
                hook_module = module
                break
        
        hook_handle = None
        if hook_module:
            hook_handle = hook_module.register_forward_hook(hook_fn)
        
        output_features = self.backbone(x, adj)
        
        if hook_handle:
            hook_handle.remove()
        
        if hidden_features is not None:
            hidden_flat = hidden_features.reshape(-1, hidden_features.shape[-1])
            if hidden_flat.shape[-1] == self.args.gcn["hidden_channel"]:
                hidden_flat = self._apply_adapters(hidden_flat, self.hidden_adapters)
        
        output_features = output_features.reshape(-1, self.args.gcn["out_channel"])
        output_features = self._apply_adapters(output_features, self.output_adapters)
        
        return output_features + data.x
    
class STLora_Model(nn.Module):
 
    def __init__(self, args):
        super(STLora_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        
        
        backbone_type = getattr(args, "backbone_type", "stgnn")
        if backbone_type == "dcrnn":
            self.backbone = DCRNN_Backbone(args)
        elif backbone_type == "astgnn":
            self.backbone = ASTGNN_Backbone(args)
        elif backbone_type == "tgcn":
            self.backbone = TGCN_Backbone(args)
        else:  
            self.backbone = STGNN_Backbone(args)
        
       
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
     
        self.lora_layers = nn.ModuleList()

        self.logger = getattr(args, "logger", None)
        if self.logger:
            self.logger.info(f"RAP initialized with backbone {backbone_type}")

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def add_lora_layer(self):
        in_dim = self.args.gcn["hidden_channel"]
        out_dim = self.args.gcn["hidden_channel"]
        lora_layer = LoRALayer(in_dim, out_dim)
        self.lora_layers.append(lora_layer)
        self.freeze_lora_layers()  
    
    def freeze_lora_layers(self):
        for lora_layer in self.lora_layers[:-1]:  
            for param in lora_layer.parameters():
                param.requires_grad = False

    def forward(self, data, adj):
        N = adj.shape[0]
        
       
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        

        feature_map = self.backbone(x, adj)
        
      
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
        
        for lora_layer in self.lora_layers:
            feature_map = lora_layer(feature_map)
        

        x = feature_map + data.x
        

        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        
     
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
   
        feature_map = self.backbone(x, adj)
        
 
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
 
        for lora_layer in self.lora_layers:
            feature_map = lora_layer(feature_map)
        

        x = feature_map + data.x
        return x


class EAC_Model(nn.Module):
 
    def __init__(self, args):
        super(EAC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank  
        
       
        backbone_type = getattr(args, "backbone_type", "stgnn")
        if backbone_type == "dcrnn":
            self.backbone = DCRNN_Backbone(args)
        elif backbone_type == "astgnn":
            self.backbone = ASTGNN_Backbone(args)
        elif backbone_type == "tgcn":
            self.backbone = TGCN_Backbone(args)
        else:  
            self.backbone = STGNN_Backbone(args)
        
        
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        
        self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))
        self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))
        
        self.year = args.year
        self.num_nodes = args.base_node_size

        self.logger = getattr(args, "logger", None)
        if self.logger:
            self.logger.info(f"RAP initialized with backbone {backbone_type}")

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
 
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
        B, N, T = x.shape
        
      
        adaptive_params = torch.mm(self.U[:N, :], self.V)  # [N, feature_dim]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]
        
      
        feature_map = self.backbone(x, adj)
        
    
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
    
        x = feature_map + data.x
        
        
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.rank, dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1))
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            self.num_nodes = new_num_nodes


class TrafficStream_Model(nn.Module):
    def __init__(self, args):
        super(TrafficStream_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        
        backbone_type = getattr(args, "backbone_type", "stgnn")
        if backbone_type == "dcrnn":
            self.backbone = DCRNN_Backbone(args)
        elif backbone_type == "astgnn":
            self.backbone = ASTGNN_Backbone(args)
        elif backbone_type == "tgcn":
            self.backbone = TGCN_Backbone(args)
        elif backbone_type == "stgnn":
            self.backbone = STGNN_Backbone(args)
 
            
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
        feature_map = self.backbone(x, adj)
        
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])  # [bs * N, feature]
        
        x = feature_map + data.x
        
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def feature(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
        feature_map = self.backbone(x, adj)
        
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])  # [bs * N, feature]
        
        x = feature_map + data.x
        
        return x


class STKEC_Model(nn.Module):

    def __init__(self, args):
        super(STKEC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        
    
        backbone_type = getattr(args, "backbone_type", "stgnn")
        if backbone_type == "dcrnn":
            self.backbone = DCRNN_Backbone(args)
        elif backbone_type == "astgnn":
            self.backbone = ASTGNN_Backbone(args)
        elif backbone_type == "tgcn":
            self.backbone = TGCN_Backbone(args)
        else:  
            self.backbone = STGNN_Backbone(args)
        

        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.ReLU()
        
      
        self.memory = nn.Parameter(torch.zeros(size=(args.cluster, args.gcn["out_channel"]), requires_grad=True))
        nn.init.xavier_uniform_(self.memory, gain=1.414)
        
        self.logger = getattr(args, "logger", None)
        if self.logger:
            self.logger.info(f"RAP initialized with backbone {backbone_type}")

    def forward(self, data, adj, scores=None):
        N = adj.shape[0]
        
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
    
        feature_map = self.backbone(x, adj)
        
  
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
      
        attention = torch.matmul(feature_map, self.memory.transpose(-1, -2)) # [bs * N, feature] * [feature , K] = [bs * N, K]
        scores = torch.nn.functional.softmax(attention, dim=1)                       # [bs * N, K]

     
        z = torch.matmul(attention, self.memory)                   # [bs * N, K] * [K, feature] = [bs * N, feature]
        
    
        x = feature_map + data.x + z
        
      
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x, scores
    
    def feature(self, data, adj, scores=None):
        N = adj.shape[0]
        
     
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
   
        feature_map = self.backbone(x, adj)
        
    
        feature_map = feature_map.reshape(-1, self.args.gcn["out_channel"])
        
   
        attention = torch.matmul(feature_map, self.memory.transpose(-1, -2)) # [bs * N, feature] * [feature , K] = [bs * N, K]

     
        z = torch.matmul(attention, self.memory)                   # [bs * N, K] * [K, feature] = [bs * N, feature]
        
        x = feature_map + data.x + z
        return x

class RAP_Model(nn.Module):
    
    def __init__(self, args):
        super(RAP_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        
        backbone_type = getattr(args, "backbone_type", "stgnn")
        if backbone_type == "dcrnn":
            self.backbone = DCRNN_Backbone(args)
        elif backbone_type == "astgnn":
            self.backbone = ASTGNN_Backbone(args)
        elif backbone_type == "tgcn":
            self.backbone = TGCN_Backbone(args)
        else:  
            self.backbone = STGNN_Backbone(args)
        
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

        self.use_strap = getattr(args, "use_strap", True)
        if self.use_strap:
            self.strap = STRAP(args)
            
            if hasattr(args, 'path'):
                self.pattern_dir = os.path.join(args.path, "pattern_libraries")
                os.makedirs(self.pattern_dir, exist_ok=True)
                

            self.strap_adapter = nn.Linear(args.gcn["out_channel"], self.strap.feature_dim)
            
            setattr(args, 'return_pattern_or_value', 'value')
        
        self.current_year = getattr(args, "year", None)
        self.pattern_initialized = False
        
        self.logger = getattr(args, "logger", None)
        if self.logger:
            msg = f"RAP initialized with backbone {backbone_type} and year {self.current_year}"
            self.logger.info(msg)
        else:
            print(f"RAP initialized with backbone {backbone_type} and year {self.current_year}")
    
    
        
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        log_fn = self.args.logger.info if hasattr(self.args, 'logger') else print
        log_fn(f"Total Parameters: {total_params}")
        log_fn(f"Trainable Parameters: {trainable_params}")
        
        if self.use_strap:
            strap_params = sum(p.numel() for p in self.strap.parameters() if p.requires_grad)
            log_fn(f"strap Parameters: {strap_params}")
    
    def initialize_patterns(self, data, adj, force=False):
        if not self.use_strap:
            return False
            
        year = self.current_year if self.current_year is not None else getattr(self.args, "year", None)
        if year is None:
            print("Year is None")
            return False
            
        has_library = self.strap.switch_to_year(year)
        
        if not has_library or force:
            print(f"Creating pattern library for year {year}...")
            success = self.strap.extract_patterns(data, adj, year)
            if success:
                self.pattern_initialized = True
                print(f"Year {year} pattern library initialized")
                return True
            else:
                print(f"Error in creating pattern library for year {year}")
                return False
        else:
            self.pattern_initialized = True
            print(f"Loaded pattern library for year {year}")
            return True
    
    def update_patterns(self, data, adj, year=None):
        if not self.use_strap:
            return False
            
        year = year or self.current_year
        if year is None:
            print("Year is None")
            return False
            
        success = self.strap.extract_patterns(data, adj, year)
        if success:
            print(f"Pattern library updated for year {year}")
            self.pattern_initialized = True
            return True
        else:
            print(f"Error in updating pattern library for year {year}")
            return False
    
    def set_year(self, year):
        self.current_year = year
        
        if self.use_strap:
            has_library = self.strap.switch_to_year(year)
            self.pattern_initialized = has_library
    
        return self.pattern_initialized
    
    def _prepare_strap(self, data, adj):
        if self.use_strap and not self.pattern_initialized and self.training:
            if hasattr(data, 'device'):
                adj_device = adj.device
                adj_cpu = adj.cpu()
                self.initialize_patterns(data, adj_cpu)
                adj = adj.to(adj_device)
            else:
                self.initialize_patterns(data, adj)
        return adj
    
    def _apply_strap(self, feature_mid):
        if not (self.use_strap and self.pattern_initialized):
            return feature_mid
            
        try:
            if self.current_year is not None:
                self.strap.switch_to_year(self.current_year)
            
            B, N, C = feature_mid.shape
            
            feature_mid_flat = feature_mid.reshape(-1, C)
            adapted_features = self.strap_adapter(feature_mid_flat)
            
            self.args.return_pattern_or_value = 'value'
            enhanced_features = self.strap(adapted_features)
            
            return enhanced_features.reshape(B, N, -1)
            
        except Exception as e:
            print(f"STRAP application error: {e}")
            import traceback
            traceback.print_exc()
            return feature_mid
    
    def forward(self, data, adj):

        adj = self._prepare_strap(data, adj)

        N = adj.shape[0]

        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        

        feature_mid = self.backbone(x, adj)

        enhanced_feature_mid = self._apply_strap(feature_mid)
        
        # 重塑特征
        feature_out = enhanced_feature_mid.reshape(-1, enhanced_feature_mid.shape[-1])
        
        if feature_out.shape[-1] != self.args.gcn["out_channel"]:
            feature_out = F.adaptive_avg_pool1d(
                feature_out.unsqueeze(1), self.args.gcn["out_channel"]
            ).squeeze(1)
        
        x = feature_out + data.x
        
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def feature(self, data, adj):
        
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        
        feature_mid = self.backbone(x, adj)
        
        enhanced_feature_mid = self._apply_strap(feature_mid)
        
        feature_out = enhanced_feature_mid.reshape(-1, enhanced_feature_mid.shape[-1])
        
        if feature_out.shape[-1] != self.args.gcn["out_channel"]:
            feature_out = F.adaptive_avg_pool1d(
                feature_out.unsqueeze(1), self.args.gcn["out_channel"]
            ).squeeze(1)
        
        x = feature_out + data.x
        return x
    
