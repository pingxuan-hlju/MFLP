import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
def roc_pr4_folder(test_x_ys, labels, pred_ys, ass_mat_shape):
    labels_mat, pred_ys_mat, test_num = torch.zeros((ass_mat_shape)) - 1, torch.zeros((ass_mat_shape)) - 1, len(labels)
    for i in range(test_num):
        labels_mat[test_x_ys[i][0]][test_x_ys[i][1]] = labels[i]
        pred_ys_mat[test_x_ys[i][0]][test_x_ys[i][1]] = pred_ys[i]
    bool_mat4test = (labels_mat != -1)
    fpr_ls, tpr_ls, recall_ls, prec_ls, effective_rows_len = [], [], [], [], 0
    for i in range(ass_mat_shape[0]):
        if (labels_mat[i][bool_mat4test[i]] == 1).sum() > 0:
            effective_rows_len += 1
            labels4test1rowi = labels_mat[i][bool_mat4test[i]]
            pred_y4test1rowi = pred_ys_mat[i][bool_mat4test[i]]
            fpr4rowi, tpr4rowi, _ = roc_curve(labels4test1rowi, pred_y4test1rowi)
            fpr_ls.append(fpr4rowi)
            tpr_ls.append(tpr4rowi)
            precision4rowi, recall4rowi, _ = precision_recall_curve(labels4test1rowi, pred_y4test1rowi)
            precision4rowi[-1] = [1, 0][precision4rowi[-2] == 0]
            prec_ls.append(precision4rowi[::-1])
            recall_ls.append(recall4rowi[::-1])

    mean_fpr, mean_recall = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    tpr_ls4mean_tpr, prec_ls4mean_prec = [], []
    for i in range(effective_rows_len):
        tpr_ls4mean_tpr.append(np.interp(mean_fpr, fpr_ls[i], tpr_ls[i]))
        prec_ls4mean_prec.append(np.interp(mean_recall, recall_ls[i], prec_ls[i]))
    mean_tpr, mean_prec = np.mean(tpr_ls4mean_tpr, axis=0), np.mean(prec_ls4mean_prec, axis=0)
    print(f'ROC平均值auc(mean_fpr, mean_tpr): {auc(mean_fpr, mean_tpr)}')
    print(f'pr平均值auc(mean_recall, mean_prec)：{auc(mean_recall, mean_prec)}')
    return mean_fpr, mean_tpr, mean_recall, mean_prec
def roc_pr4cross_val(mean_fpr_ts, mean_tpr_ts, mean_recall_ts, mean_prec_ts, k_fold):
    mean_fpr, mean_tpr, mean_recall, mean_prec = mean_fpr_ts[0], torch.mean(mean_tpr_ts, dim=0), mean_recall_ts[
        0], torch.mean(mean_prec_ts, dim=0)
    aucs4roc, aucs4pr = [], []
    for i in range(k_fold):
        aucs4roc.append(auc(mean_fpr_ts[i], mean_tpr_ts[i]))
        plt.plot(mean_fpr_ts[i], mean_tpr_ts[i], lw=1, alpha=0.3,
                 label='ROC fold %d (AUC= %0.3f)' % (i + 1, aucs4roc[i]))
    aucs4roc_std, mean_auc4roc = np.std(aucs4roc), auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8,
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc4roc, aucs4roc_std))
    plt.title('roc curve')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.axis([0, 1, 0, 1])
    plt.legend(loc='lower right')
    plt.show()
    for i in range(k_fold):
        aucs4pr.append(auc(mean_recall_ts[i], mean_prec_ts[i]))
        plt.plot(mean_recall_ts[i], mean_prec_ts[i], lw=1, alpha=0.3,
                 label='PR fold %d (AUPR= %0.3f)' % (i + 1, aucs4pr[i]))
    aucs4pr_std, mean_auc4pr = np.std(aucs4pr), auc(mean_recall, mean_prec)
    plt.plot(mean_recall, mean_prec, color='b', lw=2, alpha=0.8,
             label=r'Mean PR (AUPR = %0.3f $\pm$ %0.3f)' % (mean_auc4pr, aucs4pr_std))
    plt.title('pr curve')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.axis([0, 1, 0, 1])
    plt.legend(loc='lower right')
    plt.show()