import sys
sys.path.append('src/')
import argparse, random, re, os, shutil
import numpy as np
import sys
import json
import logging
import torch
import copy
import time
import datetime
import pytorch_influence_functions as ptif
from glob import iglob
from pathlib import Path
from datetime import datetime as dt
from scipy.stats import entropy as kldiv
from datetime import datetime
from torch_geometric.utils import to_dense_batch 
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from pytorch_influence_functions.utils import display_progress
from torch_geometric.data import Data, Dataset

from torch_geometric.data import Data, Batch, DataLoader
import torch
from scipy.spatial import distance
import os.path as osp

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
        # x: [bs, N, in_features], adj: [N, N]
        input_x = torch.matmul(adj, x)              # [N, N] * [bs, N, in_features] = [bs, N, in_features]
        output = self.weight_neigh(input_x)             # [bs, N, in_features] * [in_features, out_features] = [bs, N, out_features]
        if self.weight_self is not None:
            output += self.weight_self(x)               # [bs, N, out_features]
        return output

class Basic_model(nn.Module):
    """Some Information about Basic_Model"""
    def __init__(self,args):
        super(Basic_model, self).__init__()
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        self.memory=nn.Parameter(torch.FloatTensor(args.cluster,args.gcn["out_channel"]), requires_grad=True)
        nn.init.xavier_uniform_(self.memory, gain=1.414)
        self.args=args

    def forward(self, data, adj):
        res=data
        N = adj.shape[0]
        x = data.reshape((-1, N, 12))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        attention = torch.matmul(x,self.memory.transpose(0,1)) #B*N, memory_size #B*N, memory_size                               
        attention=F.softmax(attention,dim=1)                         
        z=torch.matmul(attention,self.memory)
        x = x + res+z
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def get_feature(data, graph, args, model, adj):
    node_size = data.shape[1]
    data = np.reshape(data[-288*7-1:-1,:], (-1, args.x_len, node_size))
    dataloader = DataLoader(continue_learning_Dataset(data), batch_size=data.shape[0], shuffle=False, pin_memory=True, num_workers=3)
    # feature shape [T', feature_dim, N]
    for data in dataloader:
        data = data.to(args.device, non_blocking=True)
        feature, _ = to_dense_batch(model.feature(data, adj), batch=data.batch)
        node_size = feature.size()[1]
        # print("before permute:", feature.size())
        feature = feature.permute(1,0,2)

        # [N, T', feature_dim]
        return feature.cpu().detach().numpy()


def get_adj(year, args):
    adj = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    return torch.from_numpy(adj).to(torch.float).to(args.device)
    

def score_func(pre_data, cur_data, args):
    # shape: [T, N]
    node_size = pre_data.shape[1]
    score = []
    for node in range(node_size):
        max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
        min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
        pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
        pre_prob = pre_prob *1.0 / sum(pre_prob)
        cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
        cur_prob = cur_prob * 1.0 /sum(cur_prob)
        score.append(kldiv(pre_prob, cur_prob))
    return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]


def influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph):
    # detect_strategy: "original": hist of original series; "feature": hist of feature at each dimension
    # pre_data/cur_data: data of seven day, shape [T, N] T=288*7
    # assert pre_data.shape[0] == 288*7
    # assert cur_data.shape[0] == 288*7
    if args.detect_strategy == 'original':
        pre_data = pre_data[-288*7-1:-1,:]
        cur_data = cur_data[-288*7-1:-1,:]
        node_size = pre_data.shape[1]
        score = []
        for node in range(node_size):
            max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
            min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
            pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
            pre_prob = pre_prob *1.0 / sum(pre_prob)
            cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
            cur_prob = cur_prob * 1.0 /sum(cur_prob)
            score.append(kldiv(pre_prob, cur_prob))
        # return staiton_id of topk max score, station with larger KL score needs more training
        return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]
    elif args.detect_strategy == 'feature':
        model.eval()
        pre_adj = get_adj(args.year-1, args)
        cur_adj = get_adj(args.year, args)
        
        pre_data = get_feature(pre_data, pre_graph, args, model, pre_adj)
        cur_data = get_feature(cur_data, cur_graph, args, model, cur_adj)
        score = []
        # print(pre_data)
        # print(pre_data.shape)
        # print(cur_data.shape)
        for i in range(pre_data.shape[0]):
            score_ = 0.0
            for j in range(pre_data.shape[2]):
                # if max(pre_data[i,:,j]) - min(pre_data[i,:,j]) == 0 and max(cur_data[i,:,j]) - min(cur_data[i,:,j]) == 0: continue
                pre_data[i,:,j] = (pre_data[i,:,j] - min(pre_data[i,:,j]))/(max(pre_data[i,:,j]) - min(pre_data[i,:,j]))
                cur_data[i,:,j] = (cur_data[i,:,j] - min(cur_data[i,:,j]))/(max(cur_data[i,:,j]) - min(cur_data[i,:,j]))
                
                pre_prob, _ = np.histogram(pre_data[i,:,j], bins=10, range=(0, 1))
                pre_prob = pre_prob *1.0 / sum(pre_prob)
                cur_prob, _ = np.histogram(cur_data[i,:,j], bins=10, range=(0, 1))
                cur_prob = cur_prob * 1.0 /sum(cur_prob)
                score_ += distance.jensenshannon(pre_prob, cur_prob)
            score.append(score_)
        # print(sorted(score))
        return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]
    else: args.logger.info("node selection mode illegal!")


def s_test(z_test, t_test,model, z_loader,  adj, gpu=-1, damp=0.01, scale=25.0, recursion_depth=5):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(z_test, t_test, model, adj,gpu)
    h_estimate = v.copy()

    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop

        for batch_idx, data in enumerate(z_loader):
            if gpu >= 0:
                x, t = data.x.to(device), data.y.to(device)
            y = model(x,adj)
            loss = calc_loss(y, t)
            params = [ p for p in model.parameters() if p.requires_grad ]
            hv = hvp(loss, params, h_estimate)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
        display_progress("Calc. s_test recursions: ", i, recursion_depth)
    return h_estimate


def calc_loss(y, t):

    # y = torch.nn.functional.log_softmax(y, dim=0)
    #y = torch.nn.functional.log_softmax(y)
    loss = torch.nn.functional.mse_loss(
        y, t,reduction='mean')
    return loss


def grad_z(z, t, model, adj,gpu=1):

    model.eval()
    # initialize

    z, t = z.to(device), t.to(device)
    y = model(z,adj)
    loss = calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    grad11=grad(loss, params, create_graph=True)
    return list(grad11)


def hvp(y, w, v):

    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads

def save_json(json_obj, json_path, append_if_exists=False, overwrite_if_exists=False, unique_fn_if_exists=True):
    """Saves a json file

    Arguments:
        json_obj: json, json object
        json_path: Path, path including the file name where the json object
            should be saved to
        append_if_exists: bool, append to the existing json file with the same
            name if it exists (keep the json structure intact)
        overwrite_if_exists: bool, xor with append, overwrites any existing
            target file
        unique_fn_if_exsists: bool, appends the current date and time to the
            file name if the target file exists already.
    """
    if isinstance(json_path, str):
        json_path = Path(json_path)

    if overwrite_if_exists:
        append_if_exists = False
        unique_fn_if_exists = False

    if unique_fn_if_exists:
        overwrite_if_exists = False
        append_if_exists = False
        if json_path.exists():
            time = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
            json_path = json_path.parents[0] / f'{str(json_path.stem)}_{time}'\
                                               f'{str(json_path.suffix)}'

    if overwrite_if_exists:
        append_if_exists = False
        with open(json_path, 'w+') as fout:
            json.dump(json_obj, fout, indent=2)
        return

    if append_if_exists:
        if json_path.exists():
            with open(json_path, 'r') as fin:
                read_file = json.load(fin)
            read_file.update(json_obj)
            with open(json_path, 'w+') as fout:
                json.dump(read_file, fout, indent=2)
            return

    with open(json_path, 'w+') as fout:
        json.dump(json_obj, fout, indent=2)


def display_progress(text, current_step, last_step, enabled=True, fix_zero_start=True):
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text) + len(str(current_step)) + len(str(last_step)) + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()


def init_logging(filename=None):
    """Initialises log/stdout output

    Arguments:
        filename: str, a filename can be set to output the log information to
            a file instead of stdout"""
    log_lvl = logging.INFO
    log_format = '%(asctime)s: %(message)s'
    if filename:
        logging.basicConfig(handlers=[logging.FileHandler(filename), logging.StreamHandler(sys.stdout)],
                            level=log_lvl,
                            format=log_format)
    else:
        logging.basicConfig(stream=sys.stdout, level=log_lvl,
                            format=log_format)


def get_default_config():
    """Returns a default config file"""
    config = {
        'outdir': 'outdir',
        'seed': 42,
        'gpu': 1,
        'dataset': 'CIFAR10',
        'num_classes': 10,
        'test_sample_num': False,
        'test_start_index': 0,
        'recursion_depth': 1,
        'r_averaging': 1,
        'scale': None,
        'damp': None,
        'calc_method': 'img_wise',
        'log_filename': None,
    }

    return config


def calc_s_test(model, test_loader, train_loader, adj,save=False, gpu=-1,
                damp=0.01, scale=25, recursion_depth=188, r=1, start=0):
    if not save:
        logging.info("ATTENTION: not saving s_test files.")
    s_tests = []
    #for i in range(start, len(test_loader.dataset)):
    for batch_idx, data in enumerate(test_loader):
        z_test, t_test = data.x,data.y
        s_test_vec = calc_s_test_single(model, z_test, t_test, train_loader,adj, gpu, damp, scale, recursion_depth, r)

        if save:
            s_test_vec = [s.cpu() for s in s_test_vec]
            torch.save(
                s_test_vec,
                os.path.join(save,(f"{batch_idx}_recdep{recursion_depth}_r{r}.s_test")))
        else:
            s_tests.append(s_test_vec)
        display_progress(
            "Calc. z_test (s_test): ", batch_idx-start, len(test_loader.dataset)-start)

    return s_tests, save


def calc_s_test_single(model, z_test, t_test, train_loader, adj,gpu=-1, damp=0.01, scale=25, recursion_depth=5000, r=1):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        z_test: test image
        t_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""
    s_test_vec_list = []
    for i in range(r):
        s_test_vec_list.append(s_test(z_test, t_test, model, train_loader,adj, gpu=gpu, damp=damp, scale=scale, recursion_depth=recursion_depth))
        display_progress("Averaging r-times: ", i, r)

    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec]

    return s_test_vec


def calc_grad_z(model, train_loader, adj,save_pth=False, gpu=-1, start=0):
    if save_pth and isinstance(save_pth, str):
        save_pth = Path(save_pth)
    if not save_pth:
        logging.info("ATTENTION: Not saving grad_z files!")

    grad_zs = []
    for idx,data in enumerate(train_loader):
        z, t = data.x,data.y
        #z = train_loader.collate_fn([z])
        #t = train_loader.collate_fn([t])
        grad_z_vec = grad_z(z, t, model, adj,gpu=gpu)
        if save_pth:
            #grad_z_vec = [g.cpu() for g in grad_z_vec]
            torch.save(grad_z_vec, save_pth.joinpath(f"{idx}.grad_z"))
        else:
            grad_zs.append(grad_z_vec)
        display_progress(
            "Calc. grad_z: ", idx-start, len(train_loader.dataset)-start)

    return grad_zs, save_pth


def load_s_test(s_test_dir, s_test_id=0, r_sample_size=10,
                train_dataset_size=-1):
    if isinstance(s_test_dir, str):
        s_test_dir = Path(s_test_dir)

    s_test = []
    logging.info(f"Loading s_test from: {s_test_dir} ...")
    num_s_test_files = 0
    a_list=[]
    for i in s_test_dir.glob("*.s_test"):
        a_list.append(i)
        num_s_test_files+=1
    if num_s_test_files != r_sample_size:
        logging.warn("Load Influence Data: number of s_test sample files ")

    for i in range(num_s_test_files):
        s_test.append(
            torch.load(a_list[i]))
        display_progress("s_test files loaded: ", i, r_sample_size)


    e_s_test = s_test[0]
    # Calculate the sum
    for i in range(len(s_test)):
        e_s_test = [i + j for i, j in zip(e_s_test, s_test[0])]

    # Calculate the average

    e_s_test = [i / len(s_test) for i in e_s_test]

    return e_s_test, s_test


def load_grad_z(grad_z_dir, train_dataset_size=-1):
    if isinstance(grad_z_dir, str):
        grad_z_dir = Path(grad_z_dir)

    grad_z_vecs = []
    logging.info(f"Loading grad_z from: {grad_z_dir} ...")
    available_grad_z_files=0
    for i in Path(args.influence_path).glob("*.grad_z"):
        available_grad_z_files+=1
    a=Path(args.influence_path).glob("*.grad_z")
    if available_grad_z_files != train_dataset_size:
        logging.warn("Load Influence Data: number of grad_z files mismatches the dataset size")
        if -1 == train_dataset_size:
            train_dataset_size = available_grad_z_files
    path=args.influence_path
    for i in range(train_dataset_size):
        grad_z_vecs.append(torch.load(path+ str(i) + ".grad_z"))
        display_progress("grad_z files loaded: ", i, train_dataset_size)

    return grad_z_vecs


def calc_influence_function(train_dataset_size, grad_z_vecs=None,
                            e_s_test=None):
    if not grad_z_vecs and not e_s_test:
        grad_z_vecs = load_grad_z()
        e_s_test, _ = load_s_test(train_dataset_size=train_dataset_size)

    if (len(grad_z_vecs) != train_dataset_size):
        logging.warn("Training data size and the number of grad_z files are inconsistent.")
        train_dataset_size = len(grad_z_vecs)

    influences = []
    for i in range(train_dataset_size):
        tmp_influence = -sum(
            [
                torch.sum(k * j).data.cpu().numpy()
                for k, j in zip(grad_z_vecs[i], e_s_test)
            ]) / train_dataset_size
        influences.append(tmp_influence)
        display_progress("Calc. influence function: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist()


def calc_influence_single(model, train_loader, test_loader, adj,test_id_num, gpu, recursion_depth, r, s_test_vec=None, time_logging=False):
    
    if not s_test_vec:
        z_test, t_test = test_loader.dataset[test_id_num]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        s_test_vec = calc_s_test_single(model, z_test, t_test, train_loader,adj, gpu, recursion_depth=recursion_depth, r=r)
    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in range(train_dataset_size):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        if time_logging:
            time_a = datetime.datetime.now()
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(f"Time for grad_z iter:"
                         f" {time_delta.total_seconds() * 1000}")
        tmp_influence = -sum(
            [
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size
        influences.append(tmp_influence)
        display_progress("Calc. influence function: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist(), test_id_num


def get_dataset_sample_ids_per_class(class_id, num_samples, test_loader, start_index=0):
    sample_list = []
    img_count = 0
    for i in range(len(test_loader.dataset)):
        _, t = test_loader.dataset[i]
        if class_id == t:
            img_count += 1
            if (img_count > start_index) and \
                    (img_count <= start_index + num_samples):
                sample_list.append(i)
            elif img_count > start_index + num_samples:
                break

    return sample_list


def get_dataset_sample_ids(num_samples, test_loader, num_classes=None, start_index=0):
    """Gets the first num_sample indices of all classes starting from
    start_index per class. Returns a list and a dict containing the indicies.

    Arguments:
        num_samples: int, number of samples of each class to return
        test_loader: DataLoader, can load the test dataset
        num_classes: int, number of classes contained in the dataset
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.

    Returns:
        sample_dict: dict, containing dict[class] = list_of_indices
        sample_list: list, containing a continious list of indices"""
    sample_dict = {}
    sample_list = []
    if not num_classes:
        num_classes = len(np.unique(test_loader.dataset.targets))
    if start_index > len(test_loader.dataset) / num_classes:
        logging.warn(f"The variable test_start_index={start_index} is  larger than the number of available samples per class.")
    for i in range(num_classes):
        sample_dict[str(i)] = get_dataset_sample_ids_per_class(
            i, num_samples, test_loader, start_index)
        # Append the new list on the same level as the old list
        # Avoids having a list of lists
        sample_list[len(sample_list):len(sample_list)] = sample_dict[str(i)]
    return sample_dict, sample_list


def load_best_model(args):
    if (args.load_first_year and args.year <= args.begin_year+1) or args.train == 0:
        load_path = args.first_year_model_path
        loss = load_path.split("/")[-1].replace(".pkl", "")
    else:
        loss = []
        for filename in os.listdir(osp.join(args.model_path, args.logname+args.time, str(args.year-1))): 
            loss.append(filename[0:-4])
        loss = sorted(loss)
        load_path = osp.join(args.model_path, args.logname+args.time, str(args.year-1), loss[0]+".pkl")
    args.logger.info("[*] load from {}".format(load_path))
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]
    model = Basic_model(args)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    return model


def calc_img_wise(config, model, train_loader, test_loader, adj):

    influences_meta = copy.deepcopy(config)
    test_sample_num = config['test_sample_num']
    test_start_index = config['test_start_index']
    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    if test_sample_num and test_start_index is not False:
        test_dataset_iter_len = test_sample_num * config['num_classes']
        _, sample_list = get_dataset_sample_ids(test_sample_num, test_loader, config['num_classes'], test_start_index)
    else:
        test_dataset_iter_len = len(test_loader.dataset)

    # Set up logging and save the metadata conf file
    logging.info(f"Running on: {test_sample_num} images per class.")
    logging.info(f"Starting at img number: {test_start_index} per class.")
    influences_meta['test_sample_index_list'] = sample_list
    influences_meta_fn = f"influences_results_meta_{test_start_index}- {test_sample_num}.json"
    influences_meta_path = outdir.joinpath(influences_meta_fn)
    save_json(influences_meta, influences_meta_path)

    influences = {}
    # Main loop for calculating the influence function one test sample per
    # iteration.
    for j in range(test_dataset_iter_len):
        # If we calculate evenly per class, choose the test img indicies
        # from the sample_list instead
        if test_sample_num and test_start_index:
            if j >= len(sample_list):
                logging.warn("ERROR: the test sample id is out of index of the defined test set. Jumping to next test sample.")
                next
            i = sample_list[j]
        else:
            i = j

        start_time = time.time()
        influence, harmful, helpful, _ = calc_influence_single(
            model, train_loader, test_loader, adj,test_id_num=i, gpu=config['gpu'],
            recursion_depth=config['recursion_depth'], r=config['r_averaging'])
        end_time = time.time()

        ###########
        # Different from `influence` above
        ###########
        influences[str(i)] = {}
        _, label = test_loader.dataset[i]
        influences[str(i)]['label'] = label
        influences[str(i)]['num_in_dataset'] = j
        influences[str(i)]['time_calc_influence_s'] = end_time - start_time
        infl = [x.cpu().numpy().tolist() for x in influence]
        influences[str(i)]['influence'] = infl
        influences[str(i)]['harmful'] = harmful[:500]
        influences[str(i)]['helpful'] = helpful[:500]

        tmp_influences_path = outdir.joinpath(f"influence_results_tmp_{test_start_index}_{test_sample_num}_last-i_{i}.json")
        save_json(influences, tmp_influences_path)
        display_progress("Test samples processed: ", j, test_dataset_iter_len)

    logging.info(f"The results for this run are:")
    logging.info("Influences: ")
    logging.info(influence[:3])
    logging.info("Most harmful img IDs: ")
    logging.info(harmful[:3])
    logging.info("Most helpful img IDs: ")
    logging.info(helpful[:3])

    influences_path = outdir.joinpath(f"influence_results_{test_start_index}_{test_sample_num}.json")
    save_json(influences, influences_path)

    return influences


def calc_all_grad_then_test(config, model, train_loader, test_loader):

    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    s_test_outdir = outdir.joinpath("s_test/")
    if not s_test_outdir.exists():
        s_test_outdir.mkdir()
    grad_z_outdir = outdir.joinpath("grad_z/")
    if not grad_z_outdir.exists():
        grad_z_outdir.mkdir()

    influence_results = {}

    calc_s_test(model, test_loader, train_loader, s_test_outdir,
                config['gpu'], config['damp'], config['scale'],
                config['recursion_depth'], config['r_averaging'],
                config['test_start_index'])
    calc_grad_z(model, train_loader, grad_z_outdir, config['gpu'],
                config['test_start_index'])

    train_dataset_len = len(train_loader.dataset)
    influences, harmful, helpful = calc_influence_function(train_dataset_len)

    influence_results['influences'] = influences
    influence_results['harmful'] = harmful
    influence_results['helpful'] = helpful
    influences_path = outdir.joinpath("influence_results.json")
    save_json(influence_results, influences_path)


def grad_z_test(z, t, model, adj,gpu=1):
    model.eval()
    # initialize
    z, t = z.to(device), t.to(device)
    y = model(z,adj)
    loss = calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    grad11=grad(loss, params, create_graph=True)
    return list(grad11)


def s_test(z_test, t_test,model, z_loader,  adj, gpu=-1, damp=0.01, scale=25.0, recursion_depth=5):
    v = grad_z_test(z_test, t_test, model, adj,gpu)
    h_estimate = v.copy()
    for i in range(recursion_depth):
        for batch_idx, data in enumerate(z_loader):
            x, t = data.x.to(device), data.y.to(device)
            y = model(x,adj)
            loss = calc_loss(y, t)
            params = [ p for p in model.parameters() if p.requires_grad ]
            hv = hvp(loss, params, h_estimate)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
        display_progress("Calc. s_test recursions: ", i, recursion_depth)
    return h_estimate


def hvp(y, w, v):
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


def calc_influence_function(train_dataset_size, save=False,grad_z_vecs=None, e_s_test=None):
    if not grad_z_vecs and not e_s_test:
        #grad_z_vecs = load_grad_z()
        #e_s_test, _ = load_s_test(train_dataset_size=train_dataset_size)
        path = save+'influence_grad_z.npy'
        a = np.load(path,allow_pickle=True)
        grad_z_vecs = a.sum(axis=0)
        save_path = save+'**.s_test'
        paths = iglob(save_path, recursive=True)
    if (len(grad_z_vecs) != train_dataset_size):
        logging.warn("Training data size and the number of grad_z files are inconsistent.")
        train_dataset_size = len(grad_z_vecs)
        
    influences_list=[]
    for path in paths:
        influences = []
        e_s_test = torch.load(path)
        for i in range(train_dataset_size):
            tmp_influence = -sum(
                [
                    torch.sum(k * j.to(device)).data.cpu().numpy()
                    for k, j in zip(grad_z_vecs[i], e_s_test)
                ]) / train_dataset_size
            influences.append(tmp_influence)
            display_progress("Calc. influence function: ", i, train_dataset_size)
        influences_list.append(influences)
    influences_np=np.array(influences_list).sum(axis=0)
    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist()


class TrafficDataset(Dataset):
    def __init__(self, inputs, split, x='', y='', edge_index='', mode='default'):
        if mode == 'default':
            self.x = inputs[split+'_x'] # [T, Len, N]
            self.y = inputs[split+'_y'] # [T, Len, N]
        else:
            self.x = x
            self.y = y
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        y = torch.Tensor(self.y[index].T)
        return Data(x=x, y=y)  
    
class continue_learning_Dataset(Dataset):
    def __init__(self, inputs):
        self.x = inputs # [T, Len, N]
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        return Data(x=x)

def influencal_node_selection(args,adj=None,inputs=None):
    global device
    device=args.device 
    pin_memory = True
    n_work = 16
    model=load_best_model(args)
    batch_size=args.batch_influ_size
    train_input=inputs['train_x'][::12,:,:]
    test_input=inputs['test_x'] 
    train_loader = DataLoader(TrafficDataset("", "", x=train_input[-args.influe_length:,:,args.past_subgraph.numpy()], y=train_input[-args.influe_length:,:,args.past_subgraph.numpy()], edge_index="", mode="subgraph"), batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=n_work)
    test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=n_work)
    grad_z_outdir = args.influence_path
    config = ptif.get_default_config()
    calc_grad_z(model, train_loader, adj,grad_z_outdir, config['gpu'], config['test_start_index'])
    calc_s_test(model,test_loader, train_loader, adj,save=grad_z_outdir, gpu=-1,damp=0.01, scale=25, recursion_depth=500, r=10, start=0)
    influences, harmful, helpful=calc_influence_function(adj.shape[0],save=grad_z_outdir)
    return helpful


def grad_z(z, t, model, adj,gpu=1):
    model.eval()
    # initialize

    z, t = z.to(device), t.to(device)
    y = model(z,adj)
    N=adj.shape[0]
    batch=10
    y=y.reshape(N,-1)
    t=t.reshape(N,-1)
    grad_list=[]
    params = [ p for p in model.parameters() if p.requires_grad]
    for i in range(N):
        loss = calc_loss(y[i], t[i])
        # Compute sum of gradients from model parameters to loss
        grad_list.append(list(grad(loss, params, create_graph=True)))
    return grad_list

def calc_grad_z(model, train_loader, adj,save_pth=False, gpu=-1, start=0):
    grad_zs = []
    for idx,data in enumerate(train_loader):
        z, t = data.x,data.y
        grad_z_vec = grad_z(z, t, model, adj,gpu=gpu)
        grad_zs.append(grad_z_vec) 
    a=np.array(grad_zs)
    path=save_pth+'influence_grad_z.npy'
    np.save(path,a)
    return grad_zs, save_pth


def get_feature(data, graph, args, model, adj):
    node_size = data.shape[1]
    data = np.reshape(data[-288*7-1:-1,:], (-1, args.x_len, node_size))
    dataloader = DataLoader(continue_learning_Dataset(data), batch_size=data.shape[0], shuffle=False, pin_memory=True, num_workers=3)
    # feature shape [T', feature_dim, N]
    for data in dataloader:
        data = data.to(args.device, non_blocking=True)
        feature, _ = to_dense_batch(model.feature(data, adj), batch=data.batch)
        node_size = feature.size()[1]
        # print("before permute:", feature.size())
        feature = feature.permute(1,0,2)

        # [N, T', feature_dim]
        return feature.cpu().detach().numpy()


def get_adj(year, args):
    adj = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    return torch.from_numpy(adj).to(torch.float).to(args.device)
    

def score_func(pre_data, cur_data, args):
    # shape: [T, N]
    node_size = pre_data.shape[1]
    score = []
    for node in range(node_size):
        max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
        min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
        pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
        pre_prob = pre_prob *1.0 / sum(pre_prob)
        cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
        cur_prob = cur_prob * 1.0 /sum(cur_prob)
        score.append(kldiv(pre_prob, cur_prob))
    # return staiton_id of topk max score, station with larger KL score needs more training
    return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]


def influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph):
    if args.detect_strategy == 'original':
        pre_data = pre_data[-288*7-1:-1,:]
        cur_data = cur_data[-288*7-1:-1,:]
        node_size = pre_data.shape[1]
        score = []
        for node in range(node_size):
            max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
            min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
            pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
            pre_prob = pre_prob *1.0 / sum(pre_prob)
            cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
            cur_prob = cur_prob * 1.0 /sum(cur_prob)
            score.append(kldiv(pre_prob, cur_prob))
        return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]
    elif args.detect_strategy == 'feature':
        model.eval()
        pre_adj = get_adj(args.year-1, args)
        cur_adj = get_adj(args.year, args)
        
        pre_data = get_feature(pre_data, pre_graph, args, model, pre_adj)
        cur_data = get_feature(cur_data, cur_graph, args, model, cur_adj)
        score = []
        for i in range(pre_data.shape[0]):
            score_ = 0.0
            for j in range(pre_data.shape[2]):
                # if max(pre_data[i,:,j]) - min(pre_data[i,:,j]) == 0 and max(cur_data[i,:,j]) - min(cur_data[i,:,j]) == 0: continue
                pre_data[i,:,j] = (pre_data[i,:,j] - min(pre_data[i,:,j]))/(max(pre_data[i,:,j]) - min(pre_data[i,:,j]))
                cur_data[i,:,j] = (cur_data[i,:,j] - min(cur_data[i,:,j]))/(max(cur_data[i,:,j]) - min(cur_data[i,:,j]))
                
                pre_prob, _ = np.histogram(pre_data[i,:,j], bins=10, range=(0, 1))
                pre_prob = pre_prob *1.0 / sum(pre_prob)
                cur_prob, _ = np.histogram(cur_data[i,:,j], bins=10, range=(0, 1))
                cur_prob = cur_prob * 1.0 /sum(cur_prob)
                score_ += distance.jensenshannon(pre_prob, cur_prob)
            score.append(score_)
        return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]
    else: args.logger.info("node selection mode illegal!")

if __name__ == "__main__":
    global device 
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--influe_length", type = int, default = 12)
    parser.add_argument("--batch_influ_size", type = int, default = 4)
    args = parser.parse_args()