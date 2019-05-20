import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

class GCN(nn.Module):
    def __init__(self, dim_nd, dim_ft, dim_hd, dim_ot, drop_rate=0.5):
        super(GCN, self).__init__()
        self.lin1 = nn.Linear(dim_ft, dim_hd)
        self.lin2 = nn.Linear(dim_hd, dim_ot)
        self.act1 = F.relu
        self.act2 = nn.Softmax
        self.drop1 = nn.Dropout(p=drop_rate)
        self.drop2 = nn.Dropout(p=drop_rate)

    def forward(self, A, X):
        temp = self.drop1(X)
        temp = torch.sparse.mm(A, temp)
        temp = self.lin1(temp)
        temp = self.act1(temp)
        temp = self.drop2(temp)
        temp = torch.sparse.mm(A, temp)
        temp = self.lin2(temp)

        output = temp
        return output

class GCN2(nn.Module):
    def __init__(self, dim_nd, dim_ft, dim_hd, dim_ot, drop_rate=0.5):
        super(GCN2, self).__init__()
        init_range = np.sqrt(6.0/(dim_ft+dim_hd))
        l1 = torch.DoubleTensor(dim_ft, dim_hd).uniform_(-init_range, init_range)
        init_range = np.sqrt(6.0/(dim_hd+dim_ot))
        l2 = torch.DoubleTensor(dim_hd, dim_ot).uniform_(-init_range, init_range)

        self.lin1 = Parameter(l1)
        self.b1 = Parameter(torch.zeros(dim_hd))
        self.lin2 = Parameter(l2)
        self.b2 = Parameter(torch.zeros(dim_ot))
        self.act1 = nn.ReLU()
        self.act2 = nn.Softmax()
        self.drop1 = nn.Dropout(p=drop_rate)
        self.drop2 = nn.Dropout(p=drop_rate)

    def forward(self, A, X):
        temp = self.drop1(X)
        temp = torch.add(torch.mm(X, self.lin1) , self.b1)
        temp = torch.sparse.mm(A, temp) 
        temp = self.act1(temp)
        temp = self.drop2(temp)
        temp = torch.add(torch.mm(temp, self.lin2) , self.b2)
        temp = torch.sparse.mm(A, temp)

        output = temp
        return output


