import sys, argparse, random, torch
sys.path.append("src/")

import numpy as np
import os.path as osp
import networkx as nx

from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph

from utils.data_convert import generate_samples
from src.model.model import TrafficStream_Model, STKEC_Model,PECPM_Model
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from model import detect_default
from src.model import replay

from utils.initialize import init, seed_anything, init_log
from utils.common_tools import mkdirs, load_best_model, long_term_pattern
from src.trainer.stkec_trainer import train, test_model


def main(args):
    args.logger.info("params : %s", vars(args))
    args.result = {"3":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "6":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "12":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "Avg":{" MAE":{}, "MAPE":{}, "RMSE":{}}}
    mkdirs(args.save_data_path)

    for year in range(args.begin_year, args.end_year+1):  # Iterate through each year from the start year to the end year
        # Loading graph data
        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        
        # Choose whether to process data or load data directly based on the data_process flag
        inputs = generate_samples(31, osp.join(args.save_data_path, str(year)), np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"], graph, val_test_mix=False) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+".npz"), allow_pickle=True)
        
        args.logger.info("[*] Year {} load from {}.npz".format(args.year, osp.join(args.save_data_path, str(year))))
        
        # Normalized adjacency matrix
        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
        
        
        if year == args.begin_year:
            long_pattern = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"][0:int(31*0.6)*288, :].reshape(int(31*0.6), 288, -1).mean(axis=1).transpose(1, 0)
            attention = long_term_pattern(args, long_pattern)
            vars(args)["attention"] = attention
            vars(args)["past_subgraph"] = np.array([i for i in range(adj.shape[0])])
        
        # If it is the first year and you need to skip the first year, the model has been trained and does not need to be retrained
        if year == args.begin_year and args.load_first_year:
            model, _ = load_best_model(args)
            test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            test_model(model, args, test_loader, pin_memory=True)
            continue
        
        # If it is an incremental strategy and the year is greater than the start year
        if year > args.begin_year and args.strategy == "incremental":
            model, _ = load_best_model(args)
            
            node_list = list()
            
            if args.increase:  # Get the newly added node
                cur_node_size = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))
            
            if args.detect:  # Get the affected nodes
                args.logger.info("[*] detect strategy {}".format(args.detect_strategy))
                pre_data = np.load(osp.join(args.raw_data_path, str(year-1)+".npz"))["x"]
                cur_data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"]
                pre_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]).edges)).T
                cur_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T
            
                old_adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
                old_adj = old_adj / (np.sum(old_adj, 1, keepdims=True) + 1e-6)
                old_adj = torch.from_numpy(old_adj).to(torch.float).to(args.device)
                
                if args.detect or args.replay:
                    influence_node_score = detect_default.influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph)
                
                if args.detect:  # Get the affected nodes
                    influence_node_list = np.argpartition(np.asarray(influence_node_score), -args.infl_topk)[-args.infl_topk:]
                    node_list.extend(list(influence_node_list))
                
                if args.replay:  # Get sampling node
                    replay_node_list = replay.replay_node_selection(args, influence_node_score, args.infl_topk)  # Select the replay node
                    node_list.extend(list(replay_node_list))
            
            
            node_list = list(set(node_list))
            if len(node_list) < int(0.1*args.graph_size):
                res=int(0.1 * args.graph_size)-len(node_list)
                res_node = [a for a in range(cur_node_size) if a not in node_list]
                expand_node_list = random.sample(res_node, res)
                node_list.extend(list(expand_node_list))
            
            
            # Get a subgraph of a node list
            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T)  # Get the index of the edge of the current year
            edge_list = list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)  # Get the list of graph edges for the current year
            
            graph_node_from_edge = set()  # Collect all nodes connected by edges
            for (u,v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            
            node_list = list(set(node_list) & graph_node_from_edge)  # Get the list of nodes in the subgraph, that is, the intersection of the nodes to be modified and the existing nodes that have changed
            
            """
            vars(args)["past_subgraph"]=args.subgraph
            """
            
            """
            If the node list is not empty
            Returns the original graph node set, the original graph edge index, and the new index of the node set used for query in the subgraph (central node set) of num_hops hops. 
            Since relabel_nodes is set to True, the nodes will be relabeled from 0, so the original graph edge index is changed to the index of the new graph.
            """
            if len(node_list) != 0:
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
                vars(args)["subgraph"] = subgraph  # Storing subgraphs
                vars(args)["subgraph_edge_index"] = subgraph_edge_index  # Store subgraph edge index
                vars(args)["mapping"] = mapping  # Storage Node Mapping
            args.logger.info("number of increase nodes:{}, nodes after {} hop:{}, total nodes this year {}".format(len(node_list), args.num_hops, args.subgraph.size(), args.graph_size))
            vars(args)["node_list"] = np.asarray(node_list)


        # When there are no nodes that need incremental training, skip this year
        if args.strategy != "retrain" and year > args.begin_year and len(args.node_list) == 0:
            model, loss = load_best_model(args)  # Load the best model
            mkdirs(osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year)))
            torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year), loss+".pkl"))
            test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            test_model(model, args, test_loader, pin_memory=True)
            args.logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
            continue
        
        if args.train:  # If training is required
            train(inputs, args)
        else:
            if args.auto_test:  # If you need automatic testing
                model, _ = load_best_model(args)
                test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
                test_model(model, args, test_loader, pin_memory=True)
    
    
    # Print different step metrics for each year
    args.logger.info("\n\n")
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            info = ""
            info_list = []
            for year in range(args.begin_year, args.end_year+1):
                if i in args.result:
                    if j in args.result[i]:
                        if year in args.result[i][j]:
                            info += "{:>10.2f}\t".format(args.result[i][j][year])
                            info_list.append(args.result[i][j][year])
            args.logger.info("{:<4}\t{}\t".format(i, j) + info + "\t{:>8.2f}".format(np.mean(info_list)))

    # Print the total training time, average training time per epoch, and number of training rounds
    total_time = 0
    for year in range(args.begin_year, args.end_year+1):
        if year in args.result:
            info = "year\t{:<4}\ttotal_time\t{:>10.4f}\taverage_time\t{:>10.4f}\tepoch\t{}".format(year, args.result[year]["total_time"], args.result[year]["average_time"], args.result[year]['epoch_num'])
            total_time += args.result[year]["total_time"]
            args.logger.info(info)
    args.logger.info("total time: {:.4f}".format(total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type = str, default = "conf/test.json")
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--backbone_type", type=str, default="stgnn", choices=["stgnn", "dcrnn", "astgnn", "tgcn"])
    parser.add_argument("--gpuid", type = int, default = 2)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--method", type = str, default = "STKEC")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--first_year_model_path", type = str, default = "log/PEMS3/trafficStream-42/2011/16.6936.pkl", help='specify a pretrained model root')
    args = parser.parse_args()
    vars(args)["device"] = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["methods"] = {'PECPM':PECPM_Model,'TrafficStream': TrafficStream_Model, 'STKEC': STKEC_Model}
    
    init(args)
    seed_anything(args.seed)
    init_log(args)
    
    main(args)