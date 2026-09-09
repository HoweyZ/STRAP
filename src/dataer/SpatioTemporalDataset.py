import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class SpatioTemporalDataset(Dataset):
    def __init__(self, inputs, split, x='', y='', edge_index='', mode='default'):
        if mode == 'default':
            x, y = inputs[split + '_x'], inputs[split + '_y']
        # Convert once per split. Each sample is a view with shape [node, step].
        self.x = torch.as_tensor(x, dtype=torch.float32).transpose(1, 2)
        self.y = torch.as_tensor(y, dtype=torch.float32).transpose(1, 2)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return Data(x=self.x[index], y=self.y[index])


class continue_learning_Dataset(Dataset):
    def __init__(self, inputs):
        self.x = torch.as_tensor(inputs, dtype=torch.float32).transpose(1, 2)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return Data(x=self.x[index])
