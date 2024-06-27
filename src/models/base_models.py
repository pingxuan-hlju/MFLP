"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args, feature_matrix, ass_mat_update1):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        self.feature_matrix = feature_matrix
        self.ass_mat_update1 = ass_mat_update1
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.cuda)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        a = manifolds
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, feature_matrix, ass_mat_update1):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(self.feature_matrix)
            feature_matrix = torch.cat([o[:, 0:1], feature_matrix], dim=1)
        h, save = self.encoder.encode(feature_matrix, self.ass_mat_update1)
        return h, save

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args, feature_matrix, ass_mat_update1):
        super(LPModel, self).__init__(args, feature_matrix, ass_mat_update1)
        self.feature_matrix = feature_matrix
        self.ass_mat_update1 = ass_mat_update1
        self.dropout = args.dropout
        self.linear2h2 = nn.Linear(123,64, bias=True )
        self.fc = nn.Sequential(
            nn.Linear(3860, 2000),
            nn.ReLU(),
            nn.Linear(2000, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 2))
        nn.init.xavier_normal_(self.fc[0].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc[2].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc[4].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc[7].weight, nn.init.calculate_gain('relu'))
    def decode(self, save, emb_all, idx):
        h0 = self.feature_matrix.to(torch.float32)
        h1 = save[0]
        h2 = save[1]
        h1_h2 = torch.cat((h1, h2), dim=1)  # 1546,128双曲的两层emb
        h =self.linear2h2(h1_h2)
        print('h1+h2:', h.shape)
        h = torch.cat((h, h0), dim=1)  # 1546, 1674 # 双曲的两层和原始特征拼接
        print('h+h0:', h.shape)
        h = torch.cat((h, emb_all), dim=1)  # 1546,1930

        emb_left, emb_right = h[idx[:, 0]], h[idx[:, 1] + 1373]
        emb= torch.cat((emb_left, emb_right), dim=1)
        return self.fc(emb.to(torch.float32))


    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])








