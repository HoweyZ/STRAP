import sys
sys.path.append('src/')
import numpy as np
from scipy.stats import entropy as kldiv
from datetime import datetime
from torch_geometric.utils import to_dense_batch 
from dataer.SpatioTemporalDataset import continue_learning_Dataset
from torch_geometric.loader import DataLoader
import torch
from scipy.spatial import distance
import os.path as osp


def get_feature(data, graph, args, model, adj):
    node_size = data.shape[1]  # Determine the number of nodes
    data = np.reshape(data[-288*7-1:-1,:], (-1, args.x_len, node_size))  # Reshape the last week’s data into a 3D array with shape (number of samples, x_len, node_size) Note: [288*7, node_size] -> [288*7/12, 12, node_size]
    dataloader = DataLoader(continue_learning_Dataset(data), batch_size=data.shape[0], shuffle=False, pin_memory=True, num_workers=32)  # Create a DataLoader object to iterate over the reshaped data. The batch_size is set to the number of samples, so one iteration is completed.
    for data in dataloader:
        data = data.to(args.device, non_blocking=True)
        feature, _ = to_dense_batch(model.feature(data, adj), batch=data.batch)  # Use the model to extract features from the data and convert it into a dense batch format, with the shape of `feature` being [batch_size, num_nodes, feature_dim]
        node_size = feature.size()[1]  #  Update node_size to match the size of the extracted features
        feature = feature.permute(1,0,2)  # Transposed feature dimensions are [num_nodes, batch_size, feature_dim]
        return feature.cpu().detach().numpy()


def get_adj(year, args):
    adj = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]  # Load an adjacency matrix from a .npz file for a specified year
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)  # Normalize the adjacency matrix by dividing each row by its sum (plus a small value to avoid division by zero)
    return torch.from_numpy(adj).to(torch.float).to(args.device)
    

def score_func(pre_data, cur_data, args):
    node_size = pre_data.shape[1]  # Determine the number of nodes
    score = []
    for node in range(node_size):
        max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))  # Find the maximum and minimum values ​​of the node in the last week period
        min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
        pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))  # Create a histogram of the data for the node in two time periods, with 10 bins in total, and normalize the histogram to get the probability distribution
        pre_prob = pre_prob *1.0 / sum(pre_prob)
        cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
        cur_prob = cur_prob * 1.0 /sum(cur_prob)
        score.append(kldiv(pre_prob, cur_prob))  # Compute the KL divergence between the two distributions and add it to the list of scores
    return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]  # Returns the indices of the top-k nodes with the highest KL divergence scores


def influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph):
    if args.detect_strategy == 'original':  # Check the detection strategy specified in the parameters
        pre_data = pre_data[-288*7-1:-1,:]  # Select the last week (7 days) of data for both datasets
        cur_data = cur_data[-288*7-1:-1,:]
        node_size = pre_data.shape[1]  # Determine the number of nodes in the data
        score = []
        for node in range(node_size):  # Iterate over each node to calculate its KL divergence score
            max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))  # Find the maximum and minimum values ​​of a node in two time periods
            min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
            pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))  # Create a histogram of the data for the node in two time periods, with 10 bins in total, and normalize the histogram to get the probability distribution
            pre_prob = pre_prob *1.0 / sum(pre_prob)
            cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
            cur_prob = cur_prob * 1.0 /sum(cur_prob)
            score.append(kldiv(pre_prob, cur_prob))  # Compute the KL divergence between the two distributions and add it to the list of scores
        return score
    elif args.detect_strategy == 'feature':
        model.eval()  # Set the model to evaluation mode
        pre_adj = get_adj(args.year-1, args)  # Get the adjacency matrix of the previous year and the current year
        cur_adj = get_adj(args.year, args)
        
        pre_data = get_feature(pre_data, pre_graph, args, model, pre_adj)   # Use the model to extract features from previous and current data, the feature dimension is [num_nodes, batch_size, feature_dim]
        cur_data = get_feature(cur_data, cur_graph, args, model, cur_adj)
        score = []
        for i in range(pre_data.shape[0]):  # Traverse the nodes in the feature data
            score_ = 0.0
            for j in range(pre_data.shape[2]):  # Traverse each feature dimension
                # if max(pre_data[i,:,j]) - min(pre_data[i,:,j]) == 0 and max(cur_data[i,:,j]) - min(cur_data[i,:,j]) == 0: continue
                pre_data[i,:,j] = (pre_data[i,:,j] - min(pre_data[i,:,j]))/(max(pre_data[i,:,j]) - min(pre_data[i,:,j]))  # Normalize the eigenvalues ​​to the range [0, 1]
                cur_data[i,:,j] = (cur_data[i,:,j] - min(cur_data[i,:,j]))/(max(cur_data[i,:,j]) - min(cur_data[i,:,j]))
                
                pre_prob, _ = np.histogram(pre_data[i,:,j], bins=10, range=(0, 1))  # Create a histogram of the distribution of the feature over two time periods, with 10 bins, and normalize the histogram to get a probability distribution
                pre_prob = pre_prob *1.0 / sum(pre_prob)
                cur_prob, _ = np.histogram(cur_data[i,:,j], bins=10, range=(0, 1))
                cur_prob = cur_prob * 1.0 /sum(cur_prob)
                score_ += distance.jensenshannon(pre_prob, cur_prob)  # Calculate the Jensen-Shannon distance between the two distributions and add to the score
            score.append(score_)  # Add the total score of the node to the score list
        return score  # Returns the indices of the top-k nodes with the highest scores
    else: args.logger.info("node selection mode illegal!")

