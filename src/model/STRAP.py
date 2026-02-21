import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import os
import pickle
import json
from datetime import datetime
from collections import Counter
import scipy.sparse as sp
from scipy import signal
import pywt 
from annoy import AnnoyIndex


class STRAP(nn.Module):
    def __init__(self, args):
        super(STRAP, self).__init__()
        self.args = args
        self.device = args.device if hasattr(args, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_dim = getattr(args.gcn, "hidden_channel", 64) if hasattr(args, 'gcn') else 64
        self.n_trees = getattr(args, "n_trees", 15)
        self.k_neighbors = getattr(args, "k_neighbors", 50) # retrieval count
        self.history_ratio = getattr(args, "history_ratio", 0.3)
        self.projection_dim = getattr(args, "projection_dim", 64)
        
        self.temporal_k_hop = getattr(args, "temporal_k_hop", 2)
        self.spatial_clusters = getattr(args, "spatial_clusters", None)
        self.spatial_retrieval_count = getattr(args, "spatial_retrieval_count", 50000)
        self.temporal_retrieval_count = getattr(args, "temporal_retrieval_count", 50000)
        self.spatiotemporal_retrieval_count = getattr(args, "spatiotemporal_retrieval_count", 100000)
        
        self.use_spatial_lib = getattr(args, "use_spatial_lib", True)
        self.use_temporal_lib = getattr(args, "use_temporal_lib", True)
        self.use_spatiotemporal_lib = getattr(args, "use_spatiotemporal_lib", True)
        
        self.spatial_dropout = getattr(args, "spatial_dropout", 0) # dropout ratio
        self.temporal_dropout = getattr(args, "temporal_dropout", 0)
        self.spatiotemporal_dropout = getattr(args, "spatiotemporal_dropout", 0)
        
        self.backbone = None
        
        self.pattern_manager = PatternLibraryManager(args)
        
        self.projector = None
        
        self.historical_patterns = {
            'spatial': [],
            'temporal': [],
            'spatiotemporal': []
        }
        
        self.historical_values = {
            'spatial': [],
            'temporal': [],
            'spatiotemporal': []
        }
        
        self.pattern_version = 0
        
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU()
        )
        
        self.pattern_embedding = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU()
        )
        
        self.fusion_weight = 0.7 # fusion ratio
        
        self.wavelet_name = 'db4'
        self.wavelet_level = 4
        
        self.topo_feature_dim = 16
        self.topo_feature_embedding = nn.Linear(self.topo_feature_dim, self.feature_dim)

        self.indices = {
            'spatial': None,
            'temporal': None,
            'spatiotemporal': None
        }
        
        self.value_indices = {
            'spatial': None,
            'temporal': None,
            'spatiotemporal': None
        }
        
        self.patterns = {
            'spatial': [],
            'temporal': [],
            'spatiotemporal': []
        }
        
        self.values = {
            'spatial': [],
            'temporal': [],
            'spatiotemporal': []
        }
        
        self.current_year = None
        self.initialized = False
        self.cached_random_patterns = self._init_random_patterns()
        
        self.current_params = {
            "n_trees": self.n_trees,
            "k_neighbors": self.k_neighbors,
            "history_ratio": self.history_ratio,
            "projection_dim": self.projection_dim,
            "temporal_k_hop": self.temporal_k_hop,
            "spatial_clusters": self.spatial_clusters,
            "spatial_retrieval_count": self.spatial_retrieval_count,
            "temporal_retrieval_count": self.temporal_retrieval_count,
            "spatiotemporal_retrieval_count": self.spatiotemporal_retrieval_count,
            "spatial_dropout": self.spatial_dropout,
            "temporal_dropout": self.temporal_dropout,
            "spatiotemporal_dropout": self.spatiotemporal_dropout
        }
    
    def set_backbone(self, backbone):
        self.backbone = backbone
    
    def _init_random_patterns(self):
        random_patterns = torch.randn(self.feature_dim, device=self.device)
        return F.normalize(random_patterns, dim=0)
    
    def _ensure_projector(self, input_dim):
        if self.projector is None:
            output_dim = min(self.projection_dim, input_dim)
            self.projector = RandomProjection(
                input_dim=input_dim,
                output_dim=output_dim,
                seed=42
            )
    
    def _build_indices(self):
        success = True
        
        for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
            use_lib = getattr(self, f"use_{pattern_type}_lib", True)
            if not use_lib:
                continue
                
            if not self.patterns[pattern_type] or len(self.patterns[pattern_type]) == 0:
                self.indices[pattern_type] = None
                success = False
                continue
                
            self.indices[pattern_type] = AnnoyIndex(self.feature_dim, 'euclidean')
            
            for i, pattern in enumerate(self.patterns[pattern_type]):
                if isinstance(pattern, torch.Tensor):
                    pattern = pattern.cpu().numpy()
                self.indices[pattern_type].add_item(i, pattern)
                
            self.indices[pattern_type].build(self.n_trees)
            
            if not self.values[pattern_type] or len(self.values[pattern_type]) == 0:
                self.value_indices[pattern_type] = None
                continue
                
            self.value_indices[pattern_type] = AnnoyIndex(self.feature_dim, 'euclidean')
            
            for i, value in enumerate(self.values[pattern_type]):
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                self.value_indices[pattern_type].add_item(i, value)
                
            self.value_indices[pattern_type].build(self.n_trees)
        
        self.initialized = success
        return success
    
    def check_params_changed(self, library_metadata):
        if not library_metadata:
            return True, {}
            
        changes = {}
        for param, value in self.current_params.items():
            if param in library_metadata and library_metadata[param] != value:
                changes[param] = {
                    "old": library_metadata[param],
                    "new": value
                }
        
        return bool(changes), changes
    
    def switch_to_year(self, year):
        if self.current_year == year and self.initialized:
            return True
        
        libraries = {}
        for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
            use_lib = getattr(self, f"use_{pattern_type}_lib", True)
            if not use_lib:
                continue
                
            library = self.pattern_manager.get_library_for_year(year, pattern_type)
            if library is not None:
                libraries[pattern_type] = library
        
        enabled_types = [t for t in ['spatial', 'temporal', 'spatiotemporal'] 
                        if getattr(self, f"use_{t}_lib", True)]
        
        if all(t in libraries for t in enabled_types):
            any_params_changed = False
            for pattern_type, library in libraries.items():
                params_changed, param_changes = self.check_params_changed(
                    library.get("metadata", {})
                )
                if params_changed:
                    any_params_changed = True
            
            for pattern_type in enabled_types:
                if pattern_type in libraries and 'patterns' in libraries[pattern_type]:
                    self.patterns[pattern_type] = libraries[pattern_type]["patterns"]
                        
                if pattern_type in libraries and 'values' in libraries[pattern_type]:
                    self.values[pattern_type] = libraries[pattern_type]["values"]
            
            self.current_year = year
            
            success = self._build_indices()
            return success
        else:
            accumulated_patterns = self._accumulate_historical_patterns(year)
            accumulated_values = self._accumulate_historical_values(year)
            
            if any(len(patterns) > 0 for t, patterns in accumulated_patterns.items() 
                if getattr(self, f"use_{t}_lib", True)):
                for pattern_type in enabled_types:
                    if pattern_type in libraries and 'patterns' in libraries[pattern_type]:
                        self.patterns[pattern_type] = libraries[pattern_type]["patterns"]
                            
                    if pattern_type in libraries and 'values' in libraries[pattern_type]:
                        self.values[pattern_type] = libraries[pattern_type]["values"]
                
                self.current_year = year
                
                for pattern_type in enabled_types:
                    if accumulated_patterns[pattern_type]:
                        metadata = {
                            "extraction_method": "Historical_Accumulation",
                            "accumulated_from": "previous_years",
                            "extraction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "pattern_version": 1,
                            **self.current_params
                        }
                        
                        library_data = {
                            "patterns": accumulated_patterns[pattern_type],
                            "values": accumulated_values[pattern_type]
                        }
                        
                        self.pattern_manager.update_library(year, library_data, metadata, pattern_type)
                
                success = self._build_indices()
                return success
        
        return False
    
    def _get_k_hop_neighbors(self, G, node, k):
        neighbors = set()
        current_layer = {node}
        
        for _ in range(k):
            next_layer = set()
            for current_node in current_layer:
                if current_node in G:
                    next_layer.update(G.neighbors(current_node))
            
            next_layer -= neighbors
            next_layer -= {node}
            
            neighbors.update(next_layer)
            
            current_layer = next_layer
            
            if not current_layer:
                break
        
        return list(neighbors)

    def _accumulate_historical_patterns(self, target_year):
        target_year = int(target_year) if isinstance(target_year, str) else target_year
        all_years = []
        
        if hasattr(self.pattern_manager, 'metadata') and 'libraries' in self.pattern_manager.metadata:
            for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
                if pattern_type in self.pattern_manager.metadata['libraries']:
                    all_years.extend([int(y) for y in self.pattern_manager.metadata['libraries'][pattern_type].keys()])
        
        all_years = list(set(all_years))
        
        if not all_years:
            return {'spatial': [], 'temporal': [], 'spatiotemporal': []}
            
        previous_years = sorted([y for y in all_years if y < target_year], reverse=True)
        if not previous_years:
            return {'spatial': [], 'temporal': [], 'spatiotemporal': []}
        
        accumulated_patterns = {'spatial': [], 'temporal': [], 'spatiotemporal': []}
        used_years = {'spatial': [], 'temporal': [], 'spatiotemporal': []}
        
        for year in previous_years:
            for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
                use_lib = getattr(self, f"use_{pattern_type}_lib", True)
                if not use_lib:
                    continue
                    
                library = self.pattern_manager.get_library_for_year(year, pattern_type)
                if library and "patterns" in library:
                    years_diff = target_year - year
                    ratio = max(0.1, 0.9 ** years_diff)
                    
                    patterns_to_take = library["patterns"]
                    
                    if len(patterns_to_take) > 10000:
                        frequencies = []
                        for pattern in patterns_to_take:
                            if isinstance(pattern, np.ndarray):
                                norm_value = np.linalg.norm(pattern)
                                frequency = np.clip(norm_value, 1e-5, 1e5) 
                            elif isinstance(pattern, torch.Tensor):
                                norm_value = torch.norm(pattern).item()
                                frequency = np.clip(norm_value, 1e-5, 1e5)
                            else:
                                frequency = 1.0
                            frequencies.append(frequency)
                        
                        frequencies = np.array(frequencies)
                        
                        min_freq = 1e-5
                        frequencies = np.maximum(frequencies, min_freq)
                        
                        frequencies_normalized = (frequencies - np.mean(frequencies)) / (np.std(frequencies) + 1e-8)
                        frequencies_normalized = np.clip(frequencies_normalized, -5, 5)
                        frequencies = np.exp(frequencies_normalized)
                        
                        inverse_importance = 1.0 / frequencies
                        inverse_importance = np.nan_to_num(inverse_importance, nan=1.0, posinf=1.0)
                        
                        inverse_sum = np.sum(inverse_importance)
                        if inverse_sum > 0:
                            sampling_probs = inverse_importance / inverse_sum
                        else:
                            sampling_probs = np.ones(len(patterns_to_take)) / len(patterns_to_take)
                        
                        sampling_probs = np.nan_to_num(sampling_probs, nan=0.0)
                        prob_sum = np.sum(sampling_probs)
                        
                        if abs(prob_sum - 1.0) > 1e-6 or np.any(sampling_probs < 0):
                            sampling_probs = np.ones(len(patterns_to_take)) / len(patterns_to_take)
                        
                        if np.any(np.isnan(sampling_probs)):
                            sampling_probs = np.ones(len(patterns_to_take)) / len(patterns_to_take)
                        
                        n_samples = max(1, int(len(patterns_to_take) * ratio))
                        
                        indices = np.random.choice(
                            len(patterns_to_take), 
                            size=n_samples, 
                            replace=False,
                            p=sampling_probs
                        )
                        patterns_to_take = [patterns_to_take[i] for i in indices]
                    
                    accumulated_patterns[pattern_type].extend(patterns_to_take)
                    used_years[pattern_type].append(year)
                    
                    if len(accumulated_patterns[pattern_type]) >= 20000:
                        break
        
        return accumulated_patterns
    
    def _accumulate_historical_values(self, target_year):
        target_year = int(target_year) if isinstance(target_year, str) else target_year
        all_years = []
        
        if hasattr(self.pattern_manager, 'metadata') and 'libraries' in self.pattern_manager.metadata:
            for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
                if pattern_type in self.pattern_manager.metadata['libraries']:
                    all_years.extend([int(y) for y in self.pattern_manager.metadata['libraries'][pattern_type].keys()])
        
        all_years = list(set(all_years))
        
        if not all_years:
            return {'spatial': [], 'temporal': [], 'spatiotemporal': []}
            
        previous_years = sorted([y for y in all_years if y < target_year], reverse=True)
        if not previous_years:
            return {'spatial': [], 'temporal': [], 'spatiotemporal': []}
        
        accumulated_values = {'spatial': [], 'temporal': [], 'spatiotemporal': []}
        used_years = {'spatial': [], 'temporal': [], 'spatiotemporal': []}
        
        for year in previous_years:
            for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
                use_lib = getattr(self, f"use_{pattern_type}_lib", True)
                if not use_lib:
                    continue
                    
                library = self.pattern_manager.get_library_for_year(year, pattern_type)
                if library and "values" in library:
                    years_diff = target_year - year
                    ratio = max(0.1, 0.9 ** years_diff)
                    
                    values_to_take = library["values"]
                    
                    if len(values_to_take) > 10000:
                        indices = np.random.choice(
                            len(values_to_take), 
                            size=int(len(values_to_take) * ratio), 
                            replace=False
                        )
                        values_to_take = [values_to_take[i] for i in indices]
                    
                    accumulated_values[pattern_type].extend(values_to_take)
                    used_years[pattern_type].append(year)
                    
                    if len(accumulated_values[pattern_type]) >= 20000:
                        break
        
        return accumulated_values
    
    def _extract_wavelet_features(self, data):
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
            
        if len(data_np.shape) > 2:
            data_np = data_np.reshape(-1, data_np.shape[-1])
            
        wavelet_features = []
        
        for i in range(data_np.shape[0]):
            node_data = data_np[i]
            
            if len(node_data) >= 2**self.wavelet_level:
                coeffs = pywt.wavedec(node_data, self.wavelet_name, level=self.wavelet_level)
                
                feature = np.concatenate([np.mean(np.abs(c)) for c in coeffs])
                
                if len(feature) > self.feature_dim:
                    feature = feature[:self.feature_dim]
                elif len(feature) < self.feature_dim:
                    padding = np.zeros(self.feature_dim - len(feature))
                    feature = np.concatenate([feature, padding])
                    
                wavelet_features.append(feature)
            else:
                wavelet_features.append(np.zeros(self.feature_dim))
                
        return np.array(wavelet_features)
    
    def _extract_network_topology_features(self, adj):
        if isinstance(adj, torch.Tensor):
            adj_np = adj.detach().cpu().numpy()
        else:
            adj_np = adj
            
        G = nx.from_numpy_array(adj_np)
        
        n_nodes = adj_np.shape[0]
        topo_features = np.zeros((n_nodes, self.topo_feature_dim))
        
        degree_centrality = nx.degree_centrality(G)
        
        clustering_coefficients = nx.clustering(G)
        
        largest_cc = max(nx.connected_components(G), key=len)
        largest_subgraph = G.subgraph(largest_cc)
        closeness_centrality = nx.closeness_centrality(largest_subgraph)
        
        if len(largest_cc) > 100:
            k = 100
            betweenness_centrality = nx.betweenness_centrality(largest_subgraph, k=k)
        else:
            betweenness_centrality = nx.betweenness_centrality(largest_subgraph)
        
        eigenvector_centrality = {i: 0.0 for i in range(n_nodes)}
        try:
            ev_centrality = nx.eigenvector_centrality_numpy(G)
            eigenvector_centrality.update(ev_centrality)
        except:
            pass
        
        for i in range(n_nodes):
            if i in G:
                topo_features[i, 0] = degree_centrality[i]
                topo_features[i, 1] = clustering_coefficients.get(i, 0)
                
                topo_features[i, 2] = closeness_centrality.get(i, 0)
                topo_features[i, 3] = betweenness_centrality.get(i, 0)
                topo_features[i, 4] = eigenvector_centrality.get(i, 0)
                
                topo_features[i, 5] = len(list(G.neighbors(i))) / n_nodes
                
                neighbor_degrees = [G.degree(neigh) for neigh in G.neighbors(i)]
                if neighbor_degrees:
                    topo_features[i, 6] = np.mean(neighbor_degrees) / n_nodes
                    topo_features[i, 7] = np.std(neighbor_degrees) / n_nodes
                
        return topo_features
    
    def chunk_graph(self, data, adj, time_window=12, overlap=6, pattern_type='spatial', 
                    k_hop=None, n_clusters=None, max_neighbors=None, max_cluster_nodes=None):
        k_hop = k_hop if k_hop is not None else self.temporal_k_hop
        max_neighbors = max_neighbors if max_neighbors is not None else 20
        max_cluster_nodes = max_cluster_nodes if max_cluster_nodes is not None else 100
        
        if isinstance(data.x, np.ndarray):
            node_features = torch.from_numpy(data.x).float()
        else:
            node_features = data.x.float()
            
        if isinstance(adj, np.ndarray):
            adj_tensor = torch.from_numpy(adj).float()
        else:
            adj_tensor = adj.float()
            
        n_nodes = adj_tensor.shape[0]
        
        if len(node_features.shape) < 2:
            raise ValueError("节点特征至少需要2个维度 [节点, 特征]")
            
        batch_size = 1
        if len(node_features.shape) > 2:
            batch_size = node_features.shape[0]
            node_features = node_features.reshape(batch_size, n_nodes, -1)
        else:
            node_features = node_features.unsqueeze(0)
            
        subgraphs = []
        
        if pattern_type == 'spatial':
            G = nx.from_numpy_array(adj_tensor.cpu().numpy())
            
            degrees = dict(G.degree())
            degree_groups = {}
            for node, degree in degrees.items():
                if degree not in degree_groups:
                    degree_groups[degree] = []
                degree_groups[degree].append(node)
            
            min_group_size = 5
            for degree, nodes in degree_groups.items():
                if len(nodes) < min_group_size:
                    continue
                    
                if len(nodes) > max_cluster_nodes:
                    for i in range(0, len(nodes), max_cluster_nodes):
                        sub_nodes = nodes[i:i+max_cluster_nodes]
                        if len(sub_nodes) >= min_group_size:
                            sub_features = node_features[:, sub_nodes]
                            sub_adj = adj_tensor[sub_nodes][:, sub_nodes]
                            
                            subgraph = {
                                'features': sub_features,
                                'adj': sub_adj,
                                'type': 'spatial',
                                'degree': degree,
                                'nodes': sub_nodes
                            }
                            subgraphs.append(subgraph)
                else:
                    sub_features = node_features[:, nodes]
                    sub_adj = adj_tensor[nodes][:, nodes]
                    
                    subgraph = {
                        'features': sub_features,
                        'adj': sub_adj,
                        'type': 'spatial',
                        'degree': degree,
                        'nodes': nodes
                    }
                    subgraphs.append(subgraph)
            
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(G)

            for comm_id, comm_nodes in enumerate(communities):
                nodes = list(comm_nodes)
                if len(nodes) < min_group_size:
                    continue
                    
                sub_features = node_features[:, nodes]
                sub_adj = adj_tensor[nodes][:, nodes]
                
                subgraph = {
                    'features': sub_features,
                    'adj': sub_adj,
                    'type': 'spatial',
                    'community_id': comm_id,
                    'nodes': nodes
                }
                subgraphs.append(subgraph)
            
        elif pattern_type == 'temporal':
            sample_size = n_nodes
            sampled_nodes = np.random.choice(n_nodes, sample_size, replace=False)
            
            for node_idx in sampled_nodes:
                G = nx.from_numpy_array(adj_tensor.cpu().numpy())
                neighbors = self._get_k_hop_neighbors(G, node_idx, k_hop)
                
                all_nodes = np.append(neighbors, node_idx)
                
                nodes_time_series = node_features[:, all_nodes, :]
                nodes_adj = adj_tensor[all_nodes][:, all_nodes]
                
                subgraph = {
                    'features': nodes_time_series,
                    'adj': nodes_adj,
                    'type': 'temporal',
                    'center_node': node_idx,
                    'nodes': all_nodes,
                    'full_time_series': True
                }
                subgraphs.append(subgraph)
        
        elif pattern_type == 'spatiotemporal':
            adj_sym = adj_tensor.cpu().numpy()
            adj_sym = (adj_sym + adj_sym.T) / 2
            
            time_windows = [(0, 1)]
            
            if batch_size > 1:
                time_windows = []
                for start_idx in range(0, batch_size, time_window - overlap):
                    end_idx = min(start_idx + time_window, batch_size)
                    if end_idx - start_idx < 2:
                        continue
                    time_windows.append((start_idx, end_idx))
                
                if not time_windows:
                    time_windows = [(0, batch_size)]
            
            for start_idx, end_idx in time_windows:
                window_features = node_features[start_idx:end_idx]
                
                if n_clusters is None:
                    n_clusters = max(2, min(5, n_nodes // 200))
                
                from sklearn.cluster import SpectralClustering
                
                adj_scipy = sp.csr_matrix(adj_sym)
                
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42
                ).fit(adj_scipy)
                
                labels = clustering.labels_
                
                for cluster_id in range(n_clusters):
                    cluster_nodes = np.where(labels == cluster_id)[0]
                    
                    if len(cluster_nodes) < 3:
                        continue
                    
                    sub_features = window_features[:, cluster_nodes]
                    sub_adj = torch.tensor(adj_sym[cluster_nodes][:, cluster_nodes], device=adj_tensor.device)
                    
                    subgraph = {
                        'features': sub_features,
                        'adj': sub_adj,
                        'type': 'spatiotemporal',
                        'window': (start_idx, end_idx),
                        'nodes': cluster_nodes,
                        'cluster_id': cluster_id
                    }
                    subgraphs.append(subgraph)
        
        return subgraphs
    
    def process_subgraph(self, subgraph, backbone=None):
        features = subgraph['features']
        adj = subgraph['adj']
        
        device = features.device
        adj = adj.to(device)
        
        if backbone is None:
            backbone = self.backbone
        
        if backbone is not None:
            value = backbone(features, adj)
            
            if len(value.shape) > 1:
                value = torch.mean(value, dim=0)
                if len(value.shape) > 1:
                    value = torch.mean(value, dim=0)
        else:
            value = torch.mean(features, dim=0)
            if len(value.shape) == 2:
                value = torch.mean(value, dim=0)
        
        key_features = []
        
        if len(features.shape) == 3:
            mean_feature = torch.mean(features, dim=0)
            
            if features.shape[0] > 1:
                std_feature = torch.std(features, dim=0, unbiased=False)
            else:
                std_feature = torch.zeros_like(mean_feature)
                
            max_feature = torch.max(features, dim=0)[0]
            min_feature = torch.min(features, dim=0)[0]
            
            mean_feature = torch.mean(mean_feature, dim=0)
            std_feature = torch.mean(std_feature, dim=0)
            max_feature = torch.mean(max_feature, dim=0)
            min_feature = torch.mean(min_feature, dim=0)
            
            key_features.append(mean_feature)
            key_features.append(std_feature)
            key_features.append(max_feature)
            key_features.append(min_feature)
        else:
            mean_feature = torch.mean(features, dim=1)
            
            if features.shape[1] > 1:
                std_feature = torch.std(features, dim=1, unbiased=False)
            else:
                std_feature = torch.zeros_like(mean_feature)
                
            max_feature = torch.max(features, dim=1)[0]
            min_feature = torch.min(features, dim=1)[0]
            
            key_features.append(mean_feature)
            key_features.append(std_feature)
            key_features.append(max_feature)
            key_features.append(min_feature)
        
        degrees = torch.sum(adj, dim=1)
        mean_degree = torch.mean(degrees)
        std_degree = torch.std(degrees)
        max_degree = torch.max(degrees)
        min_degree = torch.min(degrees)
        
        degree_feature = torch.tensor([mean_degree, std_degree, max_degree, min_degree], device=device)
        
        if degree_feature.shape[0] < self.feature_dim:
            padding = torch.zeros(self.feature_dim - degree_feature.shape[0], device=device)
            degree_feature = torch.cat([degree_feature, padding])
        else:
            degree_feature = degree_feature[:self.feature_dim]
            
        key_features.append(degree_feature)
        
        if adj.shape[0] <= 1000:
            adj_np = adj.cpu().numpy()
            G = nx.from_numpy_array(adj_np)
            
            clustering = nx.average_clustering(G)
            
            density = nx.density(G)
            
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                largest_subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(largest_subgraph)
            
            structure_feature = torch.tensor([clustering, density, avg_path_length], device=device)
            
            if structure_feature.shape[0] < self.feature_dim:
                padding = torch.zeros(self.feature_dim - structure_feature.shape[0], device=device)
                structure_feature = torch.cat([structure_feature, padding])
            else:
                structure_feature = structure_feature[:self.feature_dim]
                
            key_features.append(structure_feature)
        else:
            structure_feature = torch.zeros(self.feature_dim, device=device)
            key_features.append(structure_feature)
        
        if subgraph['type'] in ['temporal', 'spatiotemporal'] and len(features.shape) == 3:
            features_np = features.cpu().numpy()
            
            n_nodes = features_np.shape[1]
            sample_nodes = np.random.choice(n_nodes, n_nodes, replace=False)
            
            wavelet_features = []
            for node_idx in sample_nodes:
                time_series = features_np[:, node_idx, :]
                
                for dim in range(min(3, time_series.shape[1])):
                    signal = time_series[:, dim]
                    
                    if len(signal) >= 8:
                        coeffs = pywt.wavedec(signal, 'db4', level=2)
                        
                        coeff_features = []
                        for c in coeffs:
                            coeff_features.extend([np.mean(c), np.std(c), np.max(c), np.min(c)])
                        
                        if len(coeff_features) > self.feature_dim:
                            coeff_features = coeff_features[:self.feature_dim]
                        else:
                            coeff_features.extend([0] * (self.feature_dim - len(coeff_features)))
                        
                        wavelet_features.append(np.array(coeff_features))
                    else:
                        wavelet_features.append(np.zeros(self.feature_dim))
            
            if wavelet_features:
                wavelet_feature = np.mean(wavelet_features, axis=0)
                wavelet_feature_tensor = torch.tensor(wavelet_feature, device=device, dtype=torch.float32)
                key_features.append(wavelet_feature_tensor)
            else:
                key_features.append(torch.zeros(self.feature_dim, device=device))
        
        normalized_keys = []
        for key_feature in key_features:
            if key_feature.shape[0] != self.feature_dim:
                if key_feature.shape[0] > self.feature_dim:
                    key_feature = key_feature[:self.feature_dim]
                else:
                    padding = torch.zeros(self.feature_dim - key_feature.shape[0], device=device)
                    key_feature = torch.cat([key_feature, padding])
            
            key_feature = F.normalize(key_feature, p=2, dim=0)
            normalized_keys.append(key_feature)
        
        final_key = torch.mean(torch.stack(normalized_keys), dim=0)
        
        if value.shape[0] != self.feature_dim:
            if value.shape[0] > self.feature_dim:
                value = value[:self.feature_dim]
            else:
                padding = torch.zeros(self.feature_dim - value.shape[0], device=device)
                value = torch.cat([value, padding])
        
        value = F.normalize(value, p=2, dim=0)
        
        return final_key.detach(), value.detach()
    
    def extract_patterns(self, data, adj, year=None, metadata=None):
        if isinstance(adj, torch.Tensor):
            adj_np = adj.cpu().numpy()
        else:
            adj_np = adj
        
        new_patterns = {
            'spatial': [],
            'temporal': [],
            'spatiotemporal': []
        }
        
        new_values = {
            'spatial': [],
            'temporal': [],
            'spatiotemporal': []
        }
        
        for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
            use_lib = getattr(self, f"use_{pattern_type}_lib", True)
            if not use_lib:
                continue
                
            subgraphs = self.chunk_graph(
                data, 
                adj, 
                pattern_type=pattern_type, 
                k_hop=self.temporal_k_hop,
                n_clusters=self.spatial_clusters
            )
            
            for subgraph in subgraphs:
                key, value = self.process_subgraph(subgraph, self.backbone)
                
                new_patterns[pattern_type].append(key.cpu().numpy())
                new_values[pattern_type].append(value.cpu().numpy())
        
        try:
            curvature_calculator = FormanRicciCurvature(adj_np)
            spatial_patterns = curvature_calculator.identify_patterns()
            
            topo_features = self._extract_network_topology_features(adj_np)
            
            if hasattr(data, 'x') and data.x is not None:
                x = data.x
                if isinstance(x, torch.Tensor):
                    x = x.cpu().numpy()
                
                mean_feature = np.mean(x, axis=0)
                std_feature = np.std(x, axis=0)
                
                if self.use_temporal_lib and len(x.shape) > 2:
                    wavelet_features = self._extract_wavelet_features(x)
                    
                    for wf in wavelet_features[:len(wavelet_features)]:
                        temporal_feature = np.zeros(self.feature_dim)
                        temporal_feature[:min(wf.shape[0], self.feature_dim)] = wf[:min(wf.shape[0], self.feature_dim)]
                        new_patterns['temporal'].append(temporal_feature)
                        
                        if self.backbone is not None:
                            temp_x = torch.tensor(wf, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                            temp_adj = torch.eye(1, dtype=torch.float32)
                            
                            temp_value = self.backbone(temp_x, temp_adj)
                            temp_value = temp_value.squeeze().detach().cpu().numpy()
                            
                            value_feature = np.zeros(self.feature_dim)
                            value_feature[:min(temp_value.shape[0], self.feature_dim)] = temp_value[:min(temp_value.shape[0], self.feature_dim)]
                            new_values['temporal'].append(value_feature)
                        else:
                            new_values['temporal'].append(temporal_feature)
                
                if self.use_spatial_lib:
                    for node_type in ['neg_curv_nodes', 'zero_curv_nodes', 'pos_curv_nodes']:
                        for node in spatial_patterns[node_type][:len(spatial_patterns[node_type])]:
                            if node < adj_np.shape[0]:
                                node_feature = np.zeros(self.feature_dim)
                                node_feature[0] = 1.0 if node_type == 'neg_curv_nodes' else 0.0
                                node_feature[1] = 1.0 if node_type == 'zero_curv_nodes' else 0.0
                                node_feature[2] = 1.0 if node_type == 'pos_curv_nodes' else 0.0
                                
                                degree = adj_np[node].sum()
                                node_feature[3] = degree / adj_np.shape[0]
                                
                                if node < topo_features.shape[0]:
                                    topo_feat = topo_features[node]
                                    node_feature[4:4+min(topo_feat.shape[0], self.feature_dim-4)] = topo_feat[:min(topo_feat.shape[0], self.feature_dim-4)]
                                
                                new_patterns['spatial'].append(node_feature)
                                
                                if self.backbone is not None:
                                    neighbors = np.where(adj_np[node] > 0)[0]
                                    if len(neighbors) == 0:
                                        neighbors = np.array([node])
                                    
                                    sub_nodes = np.append(neighbors, node)
                                    sub_adj = adj_np[sub_nodes][:, sub_nodes]
                                    
                                    if hasattr(data, 'x') and data.x is not None:
                                        if isinstance(data.x, torch.Tensor):
                                            sub_x = data.x[sub_nodes].unsqueeze(0)
                                        else:
                                            sub_x = torch.tensor(data.x[sub_nodes], dtype=torch.float32).unsqueeze(0)
                                    else:
                                        sub_x = torch.ones((1, len(sub_nodes), self.feature_dim), dtype=torch.float32)
                                    
                                    sub_adj_tensor = torch.tensor(sub_adj, dtype=torch.float32)
                                    
                                    sub_value = self.backbone(sub_x, sub_adj_tensor)
                                    
                                    center_idx = len(sub_nodes) - 1
                                    node_value = sub_value[0, center_idx].detach().cpu().numpy()
                                    
                                    value_feature = np.zeros(self.feature_dim)
                                    value_feature[:min(node_value.shape[0], self.feature_dim)] = node_value[:min(node_value.shape[0], self.feature_dim)]
                                    new_values['spatial'].append(value_feature)
                                else:
                                    new_values['spatial'].append(node_feature)
                
                if self.use_spatiotemporal_lib and self.use_spatial_lib and self.use_temporal_lib:
                    if len(new_patterns['spatial']) > 0 and len(new_patterns['temporal']) > 0:
                        max_patterns = 2000
                        spatial_patterns = new_patterns['spatial'][:len(new_patterns['spatial'])]
                        temporal_patterns = new_patterns['temporal'][:len(new_patterns['temporal'])]
                        
                        spatial_values = new_values['spatial'][:len(new_values['spatial'])]
                        temporal_values = new_values['temporal'][:len(new_values['temporal'])]
                        
                        spatial_tensor = torch.tensor(np.array(spatial_patterns), dtype=torch.float32)
                        temporal_tensor = torch.tensor(np.array(temporal_patterns), dtype=torch.float32)
                        
                        spatial_norm = F.normalize(spatial_tensor, p=2, dim=1)
                        temporal_norm = F.normalize(temporal_tensor, p=2, dim=1)
                        
                        similarity_matrix = torch.mm(spatial_norm, temporal_norm.t())
                        
                        flat_sim = similarity_matrix.flatten()
                        _, indices = torch.topk(flat_sim, min(max_patterns, flat_sim.numel()))
                        
                        s_indices = indices // similarity_matrix.shape[1]
                        t_indices = indices % similarity_matrix.shape[1]
                        
                        for k in range(len(indices)):
                            s_idx = s_indices[k].item()
                            t_idx = t_indices[k].item()
                            
                            s_pattern = spatial_patterns[s_idx]
                            t_pattern = temporal_patterns[t_idx]
                            
                            s_pattern_tensor = torch.tensor(s_pattern, dtype=torch.float32).view(-1, 1)
                            t_pattern_tensor = torch.tensor(t_pattern, dtype=torch.float32).view(1, -1)
                            
                            cross_product = torch.matmul(s_pattern_tensor, t_pattern_tensor).flatten()
                            
                            if cross_product.size(0) > self.feature_dim:
                                pool = nn.AdaptiveAvgPool1d(self.feature_dim)
                                cross_product = pool(cross_product.unsqueeze(0)).squeeze(0)
                            elif cross_product.size(0) < self.feature_dim:
                                padding = torch.zeros(self.feature_dim - cross_product.size(0), 
                                                    dtype=torch.float32)
                                cross_product = torch.cat([cross_product, padding])
                            
                            st_pattern = F.normalize(cross_product, p=2, dim=0).cpu().numpy()
                            
                            new_patterns['spatiotemporal'].append(st_pattern)
                            
                            if s_idx < len(spatial_values) and t_idx < len(temporal_values):
                                s_value = spatial_values[s_idx]
                                t_value = temporal_values[t_idx]
                                
                                s_value_tensor = torch.tensor(s_value, dtype=torch.float32).view(-1, 1)
                                t_value_tensor = torch.tensor(t_value, dtype=torch.float32).view(1, -1)
                                
                                value_cross = torch.matmul(s_value_tensor, t_value_tensor).flatten()
                                
                                if value_cross.size(0) > self.feature_dim:
                                    pool = nn.AdaptiveAvgPool1d(self.feature_dim)
                                    value_cross = pool(value_cross.unsqueeze(0)).squeeze(0)
                                elif value_cross.size(0) < self.feature_dim:
                                    padding = torch.zeros(self.feature_dim - value_cross.size(0), 
                                                        dtype=torch.float32)
                                    value_cross = torch.cat([value_cross, padding])
                                
                                st_value = F.normalize(value_cross, p=2, dim=0).cpu().numpy()
                                
                                new_values['spatiotemporal'].append(st_value)
                            else:
                                new_values['spatiotemporal'].append(st_pattern)
        except:
            pass
        
        for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
            use_lib = getattr(self, f"use_{pattern_type}_lib", True)
            if use_lib and new_patterns[pattern_type]:
                self._store_historical_patterns(new_patterns[pattern_type], 
                                            new_values[pattern_type],
                                            pattern_type)
        
        all_patterns = {}
        all_values = {}
        for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
            use_lib = getattr(self, f"use_{pattern_type}_lib", True)
            if not use_lib:
                all_patterns[pattern_type] = []
                all_values[pattern_type] = []
                continue
                
            if self.pattern_version > 0:
                all_patterns[pattern_type], all_values[pattern_type] = self._merge_with_historical_patterns(
                    new_patterns[pattern_type], 
                    new_values[pattern_type], 
                    pattern_type
                )
            else:
                all_patterns[pattern_type] = new_patterns[pattern_type]
                all_values[pattern_type] = new_values[pattern_type]
        
        if year is not None:
            accumulated_patterns = self._accumulate_historical_patterns(year)
            accumulated_values = self._accumulate_historical_values(year)
            
            for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
                use_lib = getattr(self, f"use_{pattern_type}_lib", True)
                if not use_lib:
                    continue
                    
                if accumulated_patterns[pattern_type]:
                    if len(accumulated_patterns[pattern_type]) > 10000:
                        indices = np.random.choice(len(accumulated_patterns[pattern_type]), 10000, replace=False)
                        accumulated_patterns[pattern_type] = [accumulated_patterns[pattern_type][i] for i in indices]
                        if accumulated_values[pattern_type]:
                            accumulated_values[pattern_type] = [accumulated_values[pattern_type][i] for i in indices if i < len(accumulated_values[pattern_type])]
                    
                    all_patterns[pattern_type] = all_patterns[pattern_type] + accumulated_patterns[pattern_type]
                    
                    if accumulated_values[pattern_type]:
                        all_values[pattern_type] = all_values[pattern_type] + accumulated_values[pattern_type]
            
            for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
                use_lib = getattr(self, f"use_{pattern_type}_lib", True)
                if not use_lib:
                    continue
                    
                if metadata is None:
                    metadata = {}
                
                full_metadata = {
                    **self.current_params,
                    "extraction_method": f"{pattern_type.capitalize()}_Patterns",
                    "node_count": adj_np.shape[0],
                    "extraction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "pattern_version": self.pattern_version + 1,
                    "contains_accumulated": bool(accumulated_patterns[pattern_type])
                }
                
                full_metadata.update(metadata)
                
                library_data = {
                    "patterns": all_patterns[pattern_type],
                    "values": all_values[pattern_type]
                }
                
                success = self.pattern_manager.update_library(year, library_data, full_metadata, pattern_type)
                if success:
                    self.current_year = year
        
        self.patterns = all_patterns
        self.values = all_values
        self._build_indices()
        self.initialized = True
        self.pattern_version += 1
        
        return True
    
    def _store_historical_patterns(self, current_patterns, current_values, pattern_type):
        if not current_patterns or len(current_patterns) == 0:
            return
        
        pattern_frequencies = []
        for pattern in current_patterns:
            if isinstance(pattern, np.ndarray):
                frequency = np.linalg.norm(pattern) + 1e-6
            elif isinstance(pattern, torch.Tensor):
                frequency = torch.norm(pattern).item() + 1e-6
            else:
                frequency = 1.0
            pattern_frequencies.append(max(1e-9, frequency))
        
        inverse_importance = [1.0 / (freq + 1e-6) for freq in pattern_frequencies]
        total_inverse = sum(inverse_importance)
        sampling_probs = [imp / total_inverse for imp in inverse_importance]
        
        n_to_save = max(1, int(len(current_patterns) * self.history_ratio))
        
        indices = np.random.choice(
            len(current_patterns), 
            size=min(n_to_save, len(current_patterns)), 
            replace=False,
            p=sampling_probs
        )
        
        self.historical_patterns[pattern_type].extend([current_patterns[i] for i in indices])
        
        if current_values and len(current_values) > 0:
            for i in indices:
                if i < len(current_values):
                    self.historical_values[pattern_type].append(current_values[i])
        
        max_history = 10000
        if len(self.historical_patterns[pattern_type]) > max_history:
            hist_frequencies = []
            for pattern in self.historical_patterns[pattern_type]:
                if isinstance(pattern, np.ndarray):
                    frequency = np.linalg.norm(pattern) + 1e-6
                elif isinstance(pattern, torch.Tensor):
                    frequency = torch.norm(pattern).item() + 1e-6
                else:
                    frequency = 1.0
                hist_frequencies.append(max(1e-9, frequency))
            
            hist_inverse_importance = [1.0 / (freq + 1e-6) for freq in hist_frequencies]
            hist_total_inverse = sum(hist_inverse_importance)
            hist_probs = [imp / hist_total_inverse for imp in hist_inverse_importance]
            
            keep_indices = np.random.choice(
                len(self.historical_patterns[pattern_type]), 
                size=max_history, 
                replace=False,
                p=hist_probs
            )
            
            self.historical_patterns[pattern_type] = [self.historical_patterns[pattern_type][i] for i in keep_indices]
            
            if self.historical_values[pattern_type]:
                new_values = []
                for i in keep_indices:
                    if i < len(self.historical_values[pattern_type]):
                        new_values.append(self.historical_values[pattern_type][i])
                self.historical_values[pattern_type] = new_values
    
    def _merge_with_historical_patterns(self, current_patterns, current_values, pattern_type):
        merged_patterns = []
        merged_values = []
        
        merged_patterns.extend(current_patterns)
        
        merged_values.extend(current_values)
        
        if self.historical_patterns[pattern_type] and len(self.historical_patterns[pattern_type]) > 0:
            if hasattr(self.args, 'year') and hasattr(self.args, 'begin_year'):
                years_passed = max(0, self.args.year - self.args.begin_year)
                adaptive_ratio = max(0.1, self.history_ratio * (0.9 ** years_passed))
            else:
                adaptive_ratio = self.history_ratio
                
            hist_frequencies = []
            for pattern in self.historical_patterns[pattern_type]:
                if isinstance(pattern, np.ndarray):
                    frequency = np.linalg.norm(pattern) + 1e-6
                elif isinstance(pattern, torch.Tensor):
                    frequency = torch.norm(pattern).item() + 1e-6
                else:
                    frequency = 1.0
                hist_frequencies.append(frequency)
            
            hist_inverse_importance = [1.0 / (freq + 1e-6) for freq in hist_frequencies]
            hist_total_inverse = sum(hist_inverse_importance)
            hist_probs = [imp / hist_total_inverse for imp in hist_inverse_importance]
            
            n_historical = min(len(self.historical_patterns[pattern_type]), 
                            max(1, int(len(current_patterns) * adaptive_ratio)))
            
            indices = np.random.choice(
                len(self.historical_patterns[pattern_type]), 
                size=n_historical, 
                replace=False,
                p=hist_probs
            )
            
            merged_patterns.extend([self.historical_patterns[pattern_type][i] for i in indices])
            
            if self.historical_values[pattern_type]:
                for i in indices:
                    if i < len(self.historical_values[pattern_type]):
                        merged_values.append(self.historical_values[pattern_type][i])
                    elif merged_patterns:
                        merged_values.append(merged_patterns[-1])
        
        if len(merged_patterns) > len(merged_values):
            merged_values.extend([merged_values[-1] if merged_values else np.zeros(self.feature_dim)] * 
                                (len(merged_patterns) - len(merged_values)))
        
        return merged_patterns, merged_values
    
    def retrieve_patterns(self, x, k=None):
        if k is None:
            k = self.k_neighbors
            
        batch_size = x.shape[0]
        
        if not self.initialized or not any(self.indices.values()):
            return self.cached_random_patterns.expand(batch_size, -1)
        
        self._ensure_projector(x.shape[1])
        
        x_np = x.detach().cpu().numpy()
        x_projected = self.projector.project(x_np)
        
        retrieved_patterns = {
            'spatial': [],
            'temporal': [],
            'spatiotemporal': []
        }
        
        retrieved_values = {
            'spatial': [],
            'temporal': [],
            'spatiotemporal': []
        }
        
        dropout_stats = {
            'spatial': {'type_level': False, 'pattern_level': 0, 'total': 0},
            'temporal': {'type_level': False, 'pattern_level': 0, 'total': 0},
            'spatiotemporal': {'type_level': False, 'pattern_level': 0, 'total': 0}
        }
        
        if batch_size <= 256:
            for i in range(batch_size):
                pattern_candidates = []
                
                for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
                    use_lib = getattr(self, f"use_{pattern_type}_lib", True)
                    if not use_lib:
                        continue
                        
                    dropout_rate = getattr(self, f"{pattern_type}_dropout")
                    
                    type_level_random = np.random.random()
                    if type_level_random < dropout_rate:
                        dropout_stats[pattern_type]['type_level'] = True
                        continue
                    
                    retrieval_count = getattr(self, f"{pattern_type}_retrieval_count")
                    
                    if self.indices[pattern_type] is not None:
                        k_per_type = min(retrieval_count, len(self.patterns[pattern_type]))
                        indices, distances = self.indices[pattern_type].get_nns_by_vector(
                            x_projected[i], k_per_type, include_distances=True
                        )
                        
                        dropout_stats[pattern_type]['total'] = len(indices)
                        
                        for j, idx in enumerate(indices):
                            pattern_level_random = np.random.random()
                            if pattern_level_random < dropout_rate:
                                dropout_stats[pattern_type]['pattern_level'] += 1
                                continue
                                
                            sim_score = np.exp(-distances[j])
                            
                            pattern_entry = {
                                'pattern': self.patterns[pattern_type][idx],
                                'similarity': sim_score,
                                'type': pattern_type,
                                'index': idx
                            }
                            
                            if self.value_indices[pattern_type] is not None and idx < len(self.values[pattern_type]):
                                pattern_entry['value'] = self.values[pattern_type][idx]
                            
                            pattern_candidates.append(pattern_entry)
                
                if pattern_candidates:
                    pattern_candidates.sort(key=lambda x: x['similarity'], reverse=True)
                    selected_patterns = pattern_candidates[:min(k, len(pattern_candidates))]
                    
                    for item in selected_patterns:
                        pattern_type = item['type']
                        
                        pattern_info = {
                            'pattern': item['pattern'],
                            'similarity': item['similarity']
                        }
                        
                        if 'value' in item:
                            value_info = {
                                'value': item['value'],
                                'similarity': item['similarity']
                            }
                            retrieved_values[pattern_type].append(value_info)
                        
                        retrieved_patterns[pattern_type].append(pattern_info)
                
            final_patterns = []
            final_values = []
            
            for i in range(batch_size):
                batch_patterns = []
                batch_weights = []
                
                batch_values = []
                value_weights = []
                
                for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
                    use_lib = getattr(self, f"use_{pattern_type}_lib", True)
                    if not use_lib:
                        continue
                        
                    if i < len(retrieved_patterns[pattern_type]):
                        item = retrieved_patterns[pattern_type][i]
                        batch_patterns.append(item['pattern'])
                        batch_weights.append(item['similarity'])
                        
                    if i < len(retrieved_values[pattern_type]):
                        value_item = retrieved_values[pattern_type][i]
                        batch_values.append(value_item['value'])
                        value_weights.append(value_item['similarity'])
                
                if not batch_patterns:
                    final_patterns.append(self.cached_random_patterns)
                    final_values.append(self.cached_random_patterns)
                    continue
                
                weights = np.array(batch_weights)
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
                
                pattern = np.zeros_like(batch_patterns[0])
                for j, p in enumerate(batch_patterns):
                    pattern += weights[j] * p
                
                if batch_values:
                    value_weights_np = np.array(value_weights)
                    value_weights_np = value_weights_np / value_weights_np.sum() if value_weights_np.sum() > 0 else np.ones_like(value_weights_np) / len(value_weights_np)
                    
                    value = np.zeros_like(batch_values[0])
                    for j, v in enumerate(batch_values):
                        value += value_weights_np[j] * v
                        
                    final_values.append(torch.tensor(value, dtype=torch.float32, device=self.device))
                else:
                    final_values.append(torch.tensor(pattern, dtype=torch.float32, device=self.device))
                
                final_patterns.append(torch.tensor(pattern, dtype=torch.float32, device=self.device))
            
            if hasattr(self.args, 'return_pattern_or_value') and self.args.return_pattern_or_value == 'value':
                return torch.stack(final_values)
            else:
                return torch.stack(final_patterns)
            
        return self.cached_random_patterns.expand(batch_size, -1)
    
    def forward(self, x, year=None):
        if year is not None and year != self.current_year:
            self.switch_to_year(year)
        
        if not hasattr(self.args, 'return_pattern_or_value'):
            self.args.return_pattern_or_value = 'value'
            
        retrieved = self.retrieve_patterns(x)
        
        x_embedded = self.feature_embedding(x)
        retrieved_embedded = self.pattern_embedding(retrieved)
        
        enhanced_features = self.fusion_weight * x_embedded + (1 - self.fusion_weight) * retrieved_embedded
        
        return enhanced_features


class PatternLibraryManager:
    def __init__(self, args):
        self.args = args
        if hasattr(args, 'path'):
            self.base_path = os.path.join(args.path, "pattern_libraries")
        else:
            self.base_path = os.path.join("./data", "pattern_libraries")
            
        os.makedirs(self.base_path, exist_ok=True)
        
        self.metadata = {
            "libraries": {
                "spatial": {},
                "temporal": {},
                "spatiotemporal": {}
            },
            "last_updated": None,
            "version_history": []
        }
        
        self.current_year = None
        self.current_libraries = {
            "spatial": None,
            "temporal": None,
            "spatiotemporal": None
        }
        
        self._load_metadata()
        
    def _load_metadata(self):
        metadata_path = os.path.join(self.base_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            
            if "libraries" not in self.metadata:
                self.metadata["libraries"] = {}
            
            for pattern_type in ['spatial', 'temporal', 'spatiotemporal']:
                if pattern_type not in self.metadata["libraries"]:
                    self.metadata["libraries"][pattern_type] = {}
            
            counts = {
                pattern_type: len(libraries) 
                for pattern_type, libraries in self.metadata["libraries"].items()
            }
        else:
            self.metadata = {
                "libraries": {
                    "spatial": {},
                    "temporal": {},
                    "spatiotemporal": {}
                },
                "last_updated": None,
                "version_history": []
            }
    
    def _save_metadata(self):
        metadata_path = os.path.join(self.base_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_library_for_year(self, year, pattern_type="spatial"):
        year = str(year)
        
        if self.current_year == year and self.current_libraries[pattern_type] is not None:
            return self.current_libraries[pattern_type]
            
        if pattern_type in self.metadata["libraries"] and year in self.metadata["libraries"][pattern_type]:
            library_info = self.metadata["libraries"][pattern_type][year]
            library_path = os.path.join(self.base_path, f"{pattern_type}_library_{year}_v{library_info['version']}.pkl")
            
            if os.path.exists(library_path):
                with open(library_path, "rb") as f:
                    self.current_libraries[pattern_type] = pickle.load(f)
                self.current_year = year
                return self.current_libraries[pattern_type]
        
        return None
    
    def get_available_years(self, pattern_type="spatial"):
        available_years = []
        
        if pattern_type in self.metadata["libraries"]:
            available_years = sorted([int(y) for y in self.metadata["libraries"][pattern_type].keys()])
        
        return available_years

    def get_closest_previous_year(self, year, pattern_type="spatial"):
        year = int(year) if isinstance(year, str) else year
        available_years = self.get_available_years(pattern_type)
        
        previous_years = [y for y in available_years if y < year]
        if previous_years:
            return max(previous_years)
        return None
    
    def update_library(self, year, library_data, metadata=None, pattern_type="spatial"):
        year = str(year)
        
        version = 1
        if pattern_type in self.metadata["libraries"] and year in self.metadata["libraries"][pattern_type]:
            version = self.metadata["libraries"][pattern_type][year]["version"] + 1
        
        if isinstance(library_data, dict):
            library = library_data
            library.update({
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "version": version,
                "metadata": metadata or {}
            })
            
            if "patterns" in library:
                library["pattern_count"] = len(library["patterns"])
        else:
            library = {
                "patterns": library_data,
                "values": [],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "version": version,
                "pattern_count": len(library_data),
                "metadata": metadata or {}
            }
        
        library_path = os.path.join(self.base_path, f"{pattern_type}_library_{year}_v{version}.pkl")
        with open(library_path, "wb") as f:
            pickle.dump(library, f)
            
        if pattern_type not in self.metadata["libraries"]:
            self.metadata["libraries"][pattern_type] = {}
            
        self.metadata["libraries"][pattern_type][year] = {
            "version": version,
            "pattern_count": library.get("pattern_count", 0),
            "path": library_path,
            "created_at": library["created_at"],
            "params": metadata
        }
        
        if "values" in library:
            self.metadata["libraries"][pattern_type][year]["value_count"] = len(library["values"])
            
        self.metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["version_history"].append({
            "year": year,
            "pattern_type": pattern_type,
            "version": version,
            "timestamp": library["created_at"],
            "params": metadata
        })
        
        self._save_metadata()
        
        self.current_libraries[pattern_type] = library
        self.current_year = year
        
        return True


class FormanRicciCurvature:
    def __init__(self, adj_matrix):
        self.adj = adj_matrix
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()
            
        if adj_matrix.dtype != bool and adj_matrix.dtype != np.int64 and adj_matrix.dtype != np.int32:
            adj_matrix = (adj_matrix > 0.0).astype(np.int64)
        
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
            
        self.G = nx.from_numpy_array(adj_matrix)
                
        if not nx.is_connected(self.G):
            largest_cc = max(nx.connected_components(self.G), key=len)
            self.G = self.G.subgraph(largest_cc).copy()
        
    def compute_node_curvature(self):
        node_curvatures = np.zeros(max(self.G.nodes()) + 1)
        
        for v in self.G.nodes():
            dv = self.G.degree(v)
            if dv == 0:
                continue
                
            sum_term1 = 0
            sum_term2 = 0
            
            for u in self.G.neighbors(v):
                we = self.G[v][u].get('weight', 1.0)
                du = self.G.degree(u)
                sum_term1 += (dv * du) / (2 * we)
            
            for e in self.G.edges(v):
                wf = self.G[e[0]][e[1]].get('weight', 1.0)
                sum_term2 += dv / wf
                
            fr_curvature = 1 - sum_term1 + sum_term2
            node_curvatures[v] = fr_curvature
            
        return node_curvatures
    
    def compute_edge_curvature(self):
        edge_curvatures = {}
        
        for e in self.G.edges():
            u, v = e
            we = self.G[u][v].get('weight', 1.0)
            
            sum_term1 = 0
            sum_term2 = 0
            sum_term3 = 0
            
            for u_neighbor in self.G.neighbors(u):
                if u_neighbor != v:
                    we_prime = self.G[u][u_neighbor].get('weight', 1.0)
                    sum_term1 += we / (we * we_prime)
            
            for v_neighbor in self.G.neighbors(v):
                if v_neighbor != u:
                    we_prime = self.G[v][v_neighbor].get('weight', 1.0)
                    sum_term2 += we / (we * we_prime)
            
            for face in self._get_faces_containing_edge(e):
                wf = 1.0
                sum_term3 += we / wf
                
            fr_curvature = (we/2) - sum_term1 - sum_term2 + sum_term3
            edge_curvatures[e] = fr_curvature
            
        return edge_curvatures
    
    def _get_faces_containing_edge(self, edge):
        u, v = edge
        common_neighbors = set(self.G.neighbors(u)) & set(self.G.neighbors(v))
        faces = []
        for w in common_neighbors:
            faces.append((u, v, w))
        return faces

    def identify_patterns(self):
        node_curvatures = self.compute_node_curvature()
        edge_curvatures = self.compute_edge_curvature()
        
        neg_curv_nodes = []
        zero_curv_nodes = []
        pos_curv_nodes = []
        
        for v, c in enumerate(node_curvatures):
            if isinstance(c, np.ndarray):
                c_value = np.mean(c)
            else:
                c_value = c
                
            if c_value < 0:
                neg_curv_nodes.append(v)
            elif np.abs(c_value) < 1e-6:
                zero_curv_nodes.append(v)
            elif c_value > 0:
                pos_curv_nodes.append(v)
        
        max_nodes = 2000
        if len(neg_curv_nodes) > max_nodes:
            neg_curv_nodes = neg_curv_nodes[:max_nodes]
        if len(zero_curv_nodes) > max_nodes:
            zero_curv_nodes = zero_curv_nodes[:max_nodes]
        if len(pos_curv_nodes) > max_nodes:
            pos_curv_nodes = pos_curv_nodes[:max_nodes]
        
        high_flow_edges = []
        fluctuation_edges = []
        bottleneck_edges = []
        
        for e, c in edge_curvatures.items():
            if isinstance(c, np.ndarray):
                c_value = np.mean(c)
            else:
                c_value = c
                
            if c_value > 0.5:
                high_flow_edges.append(e)
            elif -0.5 <= c_value <= 0.5:
                fluctuation_edges.append(e)
            else:
                bottleneck_edges.append(e)
        
        return {
            'neg_curv_nodes': neg_curv_nodes,
            'zero_curv_nodes': zero_curv_nodes,
            'pos_curv_nodes': pos_curv_nodes,
            'high_flow_edges': high_flow_edges,
            'fluctuation_edges': fluctuation_edges,
            'bottleneck_edges': bottleneck_edges
        }


class RandomProjection:
    def __init__(self, input_dim, output_dim, seed=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            
        self.projection_matrix = np.random.choice([-1.0, 1.0], size=(input_dim, output_dim))
        
        self.normalization = np.sqrt(output_dim)
    
    def project(self, vectors):
        if vectors.shape[1] != self.input_dim:
            if vectors.shape[1] > self.input_dim:
                vectors = vectors[:, :self.input_dim]
            else:
                padding = np.zeros((vectors.shape[0], self.input_dim - vectors.shape[1]))
                vectors = np.concatenate([vectors, padding], axis=1)
        
        projected = np.dot(vectors, self.projection_matrix) / self.normalization
        return projected
