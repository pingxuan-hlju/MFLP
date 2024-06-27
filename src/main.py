import numpy as np
import data_loader as dl
import torch
from sklearn.model_selection import KFold
import time
from models import encoders
from tools import EarlyStopping
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
import os
from config import parser
from models.base_models import LPModel
from change_GCN import GCN
import argparse


def train(args, arg):
    torch.manual_seed(args.seed)
    epoches = args.epochs
    # batch_size, epoches = args.batchsize, args.epochs
    # 加载数据
    pos_neg_xy, pos_neg_label, rest_neg_xy, rest_neg_label = dl.gen_pos_neg_xy()
    print(
        f'pos_neg_xy: {len(pos_neg_xy)}, pos_neg_label: {len(pos_neg_label)}, rest_neg_xy: {len(rest_neg_xy)}, rest_neg_label: {len(rest_neg_label)}')
    # 加载关联矩阵、药物和微生物各自的相似性
    ass_mat1, drug_sim, micro_sim = dl.load_mats()
    kflod = KFold(n_splits=5, shuffle=False)
    for fold, (train_xy_idx, test_xy_idx) in enumerate(kflod.split(pos_neg_xy)):
        print(f'fold: {fold + 1}, 测试实例数: {len(test_xy_idx) + len(rest_neg_xy)}, 训练实例数: {len(train_xy_idx)}')
        # 对训练数据进行打包
        train_xy_label_tuple_dataset = torch.utils.data.TensorDataset(pos_neg_xy[train_xy_idx,],
                                                                      pos_neg_label[train_xy_idx,])
        train_loader = torch.utils.data.DataLoader(train_xy_label_tuple_dataset, batch_size=128, shuffle=False)
        # 对测试数据进行拼接 再打包
        test_xy_bal, test_label0_bal = pos_neg_xy[test_xy_idx,], pos_neg_label[test_xy_idx,]
        test_xy, test_label = torch.cat((test_xy_bal, rest_neg_xy), dim=0), torch.cat((test_label0_bal, rest_neg_label),
                                                                                      dim=0)
        test_xy_label_tuple_dataset = torch.utils.data.TensorDataset(test_xy, test_label)
        test_loader = torch.utils.data.DataLoader(test_xy_label_tuple_dataset, batch_size=2 * 128, shuffle=False)
        # 对关联矩阵进行遮掩
        ass_mat_update1 = ass_mat1.clone()
        ass_mat_update1[test_xy[:, 0], test_xy[:, 1]] = 0
        ass_mat_update1, drug_sim, micro_sim = ass_mat_update1.cuda(), drug_sim.cuda(), micro_sim.cuda()
        feature_matrix = torch.cat(
            (torch.cat((drug_sim, ass_mat_update1), dim=1),
             torch.cat((ass_mat_update1.T, micro_sim), dim=1))
            , dim=0)
        args.n_nodes, args.feat_dim = feature_matrix.shape
        # 获取特征矩阵
        test_xy_bal, test_label0_bal = test_xy_bal.cuda(), test_label0_bal.cuda()
        ass_mat_update1 = ass_mat_update1.cuda()
        ####################################3.27
        micro_inter = (np.loadtxt(f'../data/microbe_interactions.txt', dtype=np.int32) - 1).tolist()
        drug_inter = (np.loadtxt(f'../data/drug_interactions.txt', dtype=np.int32) - 1).tolist()
        drug_inter_mat, micro_inter_mat = np.zeros((1373, 1373)), np.zeros((173, 173))
        for item in drug_inter: drug_inter_mat[item[0], item[1]] = 1
        for item in micro_inter: micro_inter_mat[item[0], item[1]] = 1
        drug_inter_mat, micro_inter_mat = torch.from_numpy(drug_inter_mat).to(torch.long), torch.from_numpy(
            micro_inter_mat).to(torch.long)
        micro_sim[micro_inter_mat[:, 0], micro_inter_mat[:, 1]] = 1
        drug_sim[drug_inter_mat[:, 0], drug_inter_mat[:, 1]] = 1
        adj = torch.cat(
            (torch.cat((drug_sim.cuda(), ass_mat_update1), dim=1),
             torch.cat((ass_mat_update1.T, micro_sim.cuda()), dim=1))
            , dim=0)
        adj = adj.cuda()
        #
        loss_fuc = nn.CrossEntropyLoss()
        # 初始化模型
        model = LPModel(args, feature_matrix, adj)
        model2 = GCN(arg)
        model2 = model2.cuda()
        # print(model)
        # model = model.encode(feature_matrix, ass_mat1)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        early_stopping = EarlyStopping(patience=60, verbose=True, save_path=f'../pt/checkpoint.pt')
        model = model.cuda()
        model.train()
        for epoch in range(epoches):
            for step, (xy4train_batch, label4train_batch) in enumerate(train_loader):
                # 训练
                model.train()
                xy4train_batch, label4train_batch = xy4train_batch.cuda(), label4train_batch.cuda()
                t_start = time.time()
                embeddings, save = model.encode(feature_matrix, adj)
                # input = (drug_sim, micro_sim, hete_ass_idx, hete_ass_weight, feature_matrix)
                _, train_emb_all = model2(xy4train_batch[:, 0], (xy4train_batch[:, 1]), feature_matrix, adj, arg)
                outs = model.decode(save, train_emb_all, xy4train_batch)
                # label4train_batch = label4train_batch.unsqueeze(1)
                t_end = time.time()
                train_loss = loss_fuc(outs, label4train_batch.to(torch.long))
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                correct_num = (torch.max(outs, dim=1)[1] == label4train_batch).float().sum()
                print(
                    f'epoch: {epoch + 1}, step: {step + 1}, train loss: {train_loss.item()}, time: {t_end - t_start}, acc: {correct_num / len(label4train_batch)}')
                model.eval()
                with torch.no_grad():
                    t_start = time.time()
                    embeddings, save = model.encode(feature_matrix, adj)
                    _, test_emb_all = model2(test_xy_bal[:, 0], test_xy_bal[:, 1], feature_matrix, adj, arg)
                    outs = model.decode(save, test_emb_all, test_xy_bal)
                    t_end = time.time()
                    test_loss = loss_fuc(outs, test_label0_bal.to(torch.long))
                    correct_num = (torch.max(outs, dim=1)[1] == test_label0_bal).float().sum()
                    print(
                        f'epoch: {epoch + 1}, step: {step + 1}, test loss: {test_loss.item()}, time: {t_end - t_start}, acc: {correct_num / len(test_label0_bal)}')
                    early_stopping(test_loss, model)
                if early_stopping.early_stop:
                    print(f'early_stopping!')
                    break
            if early_stopping.early_stop:
                print(f'early_stopping!')
                break
        model.load_state_dict(torch.load(os.path.join('..', 'pt', 'checkpoint.pt')))
        model.eval()
        with torch.no_grad():
            pred_ls = []
            for step, (x_y, x_y_label) in enumerate(test_loader):
                embeddings, save = model.encode(feature_matrix, adj)
                _, end_test_emb_all = model2(x_y[:, 0], x_y[:, 1], feature_matrix, adj, arg)
                x_y, x_y_label = x_y, x_y_label
                outs = model.decode(save, end_test_emb_all, x_y)
                pred_ls.append(outs)
            preds = torch.cat(pred_ls, dim=0)
            preds, test_label = preds.cuda(), test_label.cuda()
            correct_num = (torch.max(preds, dim=1)[1] == test_label).float().sum()
            acc = correct_num / (len(test_label))
            precision, recall, threshold = precision_recall_curve(test_label.cpu(), preds[:, 1].cpu())
            roc_auc, aupr = roc_auc_score(test_label.cpu(), preds[:, 1].cpu()), auc(recall, precision)
            print(f'acc: {acc}, roc_auc值: {roc_auc}, aupr值: {aupr}')
            dl.outs2file(fold, preds[:, 1].cpu(), test_label.cpu(), test_xy)
            dl.gen_file_for_args(args, acc, roc_auc, aupr)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='GCN')
    ap.add_argument('--drug_nums', type=int, default=1373)
    ap.add_argument('--microbe_nums', type=int, default=173)
    ap.add_argument('--layer1_hidden_units', type=int, default=64)
    ap.add_argument('--layer2_hidden_units', type=int, default=64)
    ap.add_argument('--k_fold', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--seed', type=int, default=1206)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--threshold', type=float, default=0.9)
    arg = ap.parse_args([])
    args = parser.parse_args()
    for lr in [3e-5]:
        for dropout in [0.2]:
            for c in [None]:
                args.lr = lr
                arg.lr = lr
                args.c = c
                args.dropout = dropout
                train(args, arg)
