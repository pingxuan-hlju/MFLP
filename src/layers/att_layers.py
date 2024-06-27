import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, in_features, bias=True)
        self.linear1 = nn.Linear(in_features, 1, bias=True)
        self.in_features = in_features

    def forward (self, x, adj):
        n = x.size(0)
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)
        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat)
        att_adj = self.linear1(att_adj).squeeze()
        att_adj = F.sigmoid(att_adj)
        att_adj = torch.mul(adj, att_adj)
        return att_adj