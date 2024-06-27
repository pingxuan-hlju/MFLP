##############################################
import data_loader as dl
import argparse
from sklearn.model_selection import KFold
import torch
from tools import EarlyStopping
from time import time
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
import torch.nn as nn
from torch_geometric.nn import conv


class GCN(nn.Module):
    # 参数对象; 药物相似性; 微生物相似性; 遮掩的关联矩阵; 特征矩阵;
    def __init__(self, args):
        super().__init__()
        self.drug_nums, self.micro_nums = args.drug_nums, args.microbe_nums
        self.gcn1_units, self.gcn2_units = args.layer1_hidden_units, args.layer2_hidden_units
        # self.kernel_size = args.gcn_layer_num + 1
        gcn1_innum = 256
        self.gcn1 = conv.GCNConv(gcn1_innum, self.gcn1_units)
        self.gcn2 = conv.GCNConv(self.gcn1_units, self.gcn2_units)
        self.gcn3 = conv.GCNConv(gcn1_innum, self.gcn1_units)
        self.gcn4 = conv.GCNConv(self.gcn1_units, self.gcn2_units)
        self.gcn5 = conv.GCNConv(gcn1_innum, self.gcn1_units)
        self.gcn6 = conv.GCNConv(self.gcn1_units, self.gcn2_units)
        self.cnn = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=1, dilation=2, padding=2)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2))
        nn.init.xavier_normal_(self.fc[0].weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc[3].weight, nn.init.calculate_gain('relu'))
        self.fc1 = nn.Sequential(
            nn.Linear(1546, 256),
            nn.ReLU()
        )
        nn.init.xavier_normal_(self.fc1[0].weight, nn.init.calculate_gain('relu'))

    # X, 特征矩阵; ass_mat, 遮掩之后的关联矩阵;
    def forward(self, left, right, X, adj, arg):
        # gcn1, emb1, (128, 128);
        ########L=2########################
        adj_inf = dl.compute_inf(adj.cpu(), L=2)
        adj_inf = adj_inf.cuda()
        hete_ass_idx = torch.nonzero(adj_inf > arg.threshold).to(torch.long)
        # 获取异质网络对应边的权重, (235078,)
        hete_ass_weight = adj[hete_ass_idx[:, 0], hete_ass_idx[:, 1]]
        # x_fea = X
        X = self.fc1(X.to(torch.float32))
        emb1 = self.relu(self.gcn1(X, hete_ass_idx.T, hete_ass_weight)).to(torch.float32)
        emb2 = self.relu(self.gcn2(emb1, hete_ass_idx.T, hete_ass_weight)).to(torch.float32)
        emb_l2 = torch.cat((emb1, emb2), dim=1)
        #########################################
        ###########l=1#####################
        adj_inf = dl.compute_inf(adj.cpu(), L=1)
        adj_inf = adj_inf.cuda()
        hete_ass_idx = torch.nonzero(adj_inf > arg.threshold).to(torch.long)
        # 获取异质网络对应边的权重, (235078,)
        hete_ass_weight = adj[hete_ass_idx[:, 0], hete_ass_idx[:, 1]]
        emb3 = self.relu(self.gcn3(X, hete_ass_idx.T, hete_ass_weight)).to(torch.float32)
        emb4 = self.relu(self.gcn4(emb3, hete_ass_idx.T, hete_ass_weight)).to(torch.float32)
        emb_l1 = torch.cat((emb3, emb4), dim=1)
        ###############原始A######################################
        hete_ass_idx = torch.nonzero(adj > arg.threshold).to(torch.long)
        # 获取异质网络对应边的权重, (235078,)
        hete_ass_weight = adj[hete_ass_idx[:, 0], hete_ass_idx[:, 1]]
        emb5 = self.relu(self.gcn5(X, hete_ass_idx.T, hete_ass_weight)).to(torch.float32)
        emb6 = self.relu(self.gcn6(emb5, hete_ass_idx.T, hete_ass_weight)).to(torch.float32)
        emb_l0 = torch.cat((emb5, emb6), dim=1)
        ###########下面应该做空洞卷积将三个emb合成一个######################################
        emb = torch.stack((emb_l0,emb_l1,emb_l2),dim=0).unsqueeze(0)
        emb_cnn = self.cnn(emb)
        emb_all = emb_cnn.squeeze(0).squeeze(0)
        #############################################
        emb_all_fea = emb_all
        # danlu GCN
        emb_left, emb_right = emb_all_fea[left], emb_all_fea[right]
        emb = torch.cat((emb_left, emb_right), dim=1)
        emb1 = self.fc(emb)
        return emb1, emb_all
