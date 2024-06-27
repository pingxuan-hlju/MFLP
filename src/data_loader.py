import os

from scipy.io import loadmat
import torch
import numpy as np
import math
import torch.nn.functional as F


# @ 加载矩阵
def load_mats(dir='../data/'):
    ass_matrix = torch.from_numpy(np.loadtxt(f'{dir}interaction.txt'))
    # 加载通过疾病过渡的邻接矩阵
    drug_sim = torch.from_numpy(np.loadtxt(f'{dir}drugsimilarity.txt'))
    micro_sim = torch.from_numpy(np.loadtxt(f'{dir}microbe_microbe_similarity.txt'))
    return ass_matrix, drug_sim, micro_sim


# @ 生成训练集、测试集
def gen_pos_neg_xy(dir='../data/'):
    ass_mat, rand_num_4940 = torch.from_numpy(np.loadtxt(f'{dir}interaction.txt')).to(
        torch.float), torch.randperm(4940)
    pos_xy = ass_mat.nonzero(as_tuple=False)
    pos_label = torch.ones(len(pos_xy))
    neg_xy = tensor_shuffle((ass_mat == 0).nonzero(as_tuple=False), dim=0)
    neg_xy, rest_neg_xy, neg_label, rest_neg_label = neg_xy[0: len(pos_xy)], neg_xy[len(pos_xy):], torch.zeros(
        len(pos_xy)), torch.zeros(len(neg_xy) - len(pos_xy))
    pos_neg_xy, pos_neg_label, rest_neg_xy, rest_neg_label = torch.cat((pos_xy, neg_xy), dim=0), torch.cat(
        (pos_label, neg_label), dim=0), rest_neg_xy, rest_neg_label
    pos_neg_xy, pos_neg_label = pos_neg_xy[rand_num_4940], pos_neg_label[rand_num_4940]
    return pos_neg_xy, pos_neg_label.to(torch.float), rest_neg_xy, rest_neg_label.to(torch.float)


# @ tensor 版本的shuffle 按维度0
def tensor_shuffle(ts, dim=0):
    return ts[torch.randperm(ts.shape[dim])]


# @ 计算高斯核相似性
def gauss_sim(mat):
    # 高斯核
    sigma = 1 / torch.diag(torch.matmul(mat, mat.T)).mean()
    # 向量写法
    sim_mat = torch.mul(mat, mat).sum(dim=1, keepdims=True) + torch.mul(mat, mat).sum(dim=1,
                                                                                      keepdims=True).T - 2 * torch.matmul(
        mat, mat.T)
    # 返回高斯核相似性矩阵
    sim_mat = torch.exp(-1 * sigma * sim_mat)
    # MTK...
    # sim_mat= 1/ (1+ torch.exp(-15* sim_mat+ math.log(9999)))
    return sim_mat


# @ 计算hamming interaction profile similarity.
def hip_sim(mat):
    sim_ls, dim = [], mat.shape[1]
    for i in range(mat.shape[0]):
        sim_ls.append(((mat[i] - mat) == 0).sum(dim=1) / dim)
    return torch.stack(sim_ls)


# @ 带重启的随机游走, mat, like, (1373, 1373)
def rwr(mat, times=10):
    # 0.1概率继续走, 0.9的概率回初始状态
    alpha = 0.1
    # 行归一化
    trans_mat = mat / (mat.sum(dim=1, keepdims=True) + 1e-15)
    state_mat = torch.eye(mat.shape[0])
    # 游走
    for i in range(times):
        state_mat = alpha * torch.matmul(trans_mat, state_mat) + (1 - alpha) * torch.eye(mat.shape[0])
    return state_mat

def gen_file_for_args(args, acc, auc, aupr):
    file_path = './eva/eva.txt'
    with open(file_path, 'a') as f:
        f.write(
            f'{args.lr}\t{args.dropout}\t{args.c}\t{acc}\t{auc}\t{aupr}\n')

def IGCNgen_file_for_args(args, acc, auc, aupr):
    file_path = './eva-GCN/eva.txt'
    with open(file_path, 'a') as f:
        f.write(
            f'{args.lr}\t{args.epochs}\t{args.threshold}\t{acc}\t{auc}\t{aupr}\n')


def compute_inf(adj, L):
    c = 0.1
    W = torch.zeros(adj.shape)
    D = (torch.diag(adj.sum(dim=1)) + 1e-15) ** (-1)
    for gamma in range(L):
        W += c * ((1 - c) ** gamma) * ((D * adj) ** gamma)
    W = torch.mul(W, adj > 0)
    W = torch.softmax(W, dim=1)
    return W

# 写入文件
def outs2file(fold, preds, labels, test_xy):
    eva_labels_outs_x_y = np.zeros((len(test_xy), 4))
    for i in range(len(test_xy)):
        eva_labels_outs_x_y[i, 2], eva_labels_outs_x_y[i, 3], = test_xy[i, 0], test_xy[i, 1]
        eva_labels_outs_x_y[i, 0] = labels[i]
        eva_labels_outs_x_y[i, 1] = preds[i]
    np.savetxt(f'./GCN-result/fold{fold}.txt', eva_labels_outs_x_y)
