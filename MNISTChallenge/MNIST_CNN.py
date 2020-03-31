import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import torch

torch.set_default_dtype(torch.float32)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        // ....
        pass

    def forward(self, x):
        pass

    def train(self,dataset):
        //...
        pass

    def test(self,dataset):
        //...
        pass



    
