##########################################################################
#
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
#
# @modify : chri0220@skku.edu, ghkim10202@skku.edu
# add fairness scores 
#
##########################################################################

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

import numpy as np
from scipy import stats
import networkx as nx

def get_accuracy_scores(grt, y_pred, median=0.5):
    # check if -1 is used for negatives label instead of 0.
    idx = grt[:, 2] == -1
    grt[idx, 2] = 0
    # create the acctual labels vector
    y_true = grt[:, 2]

    # replace inf to max num
    replace_val = np.nan_to_num(np.inf)
    y_pred[np.isfinite(y_pred) == False] = replace_val

    # calcualte the accuracy
    acc = accuracy_score(y_true, y_pred > median)
    roc_score = roc_auc_score(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_pred)

    return acc, roc_score, ap_score


def get_fairness_scores(G, communities, test, y_pred, median=0.5):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    test_pred_total = np.hstack((test, y_pred.reshape(-1, 1)))
    test_edges_true = test_pred_total[test_pred_total[:, 2] == 1]
    test_edges_false = test_pred_total[test_pred_total[:, 2] == 0]

    sensitive = np.zeros(G.number_of_nodes())
    for i in range(G.number_of_nodes()):
        for g in range(len(communities)):
            if i in communities[g]:
                sensitive[i] = g

    preds_pos_intra = []
    preds_pos_inter = []
    for e in test_edges_true:
        if sensitive[int(e[0])] == sensitive[int(e[1])]:
            preds_pos_intra.append(sigmoid(e[3]))
        else:
            preds_pos_inter.append(sigmoid(e[3]))

    preds_neg_intra = []
    preds_neg_inter = []
    for e in test_edges_false:
        if sensitive[int(e[0])] == sensitive[int(e[1])]:
            preds_neg_intra.append(sigmoid(e[3]))
        else:
            preds_neg_inter.append(sigmoid(e[3]))

    res = {}
    for preds_pos, preds_neg, type in zip((preds_pos_intra, preds_pos_inter),
                                          (preds_neg_intra, preds_neg_inter),
                                          ("intra", "inter")):
        err = (np.sum(list(map(lambda x: x >= median, preds_pos))) +
               np.sum(list(map(lambda x: x < median, preds_neg)))) / (len(preds_pos) + len(preds_neg))

        score_avg = (sum(preds_pos) + sum(preds_neg)) / (len(preds_pos) + len(preds_neg))
        pos_avg, neg_avg = sum(preds_pos) / len(preds_pos), sum(preds_neg) / len(preds_neg)

        res[type] = [err, score_avg, pos_avg, neg_avg]

    ks_pos = stats.ks_2samp(preds_pos_intra, preds_pos_inter)[0]
    ks_neg = stats.ks_2samp(preds_neg_intra, preds_neg_inter)[0]
    scores = [abs(res["intra"][i] - res["inter"][i]) for i in range(1, 4)] + [ks_pos, ks_neg]

    return scores


def get_modularity(G, y_pred, communities, grt, median=0.5):
    G_ground = G.copy()
    G_new = G.copy()
    for i in range(grt.shape[0]):
        if y_pred[i] > median:
            G_new.add_edge(grt[i][0], grt[i][1])
        if grt[i][2] == 1:
            G_ground.add_edge(grt[i][0], grt[i][1])
    modularity_new = nx.community.modularity(G_new, communities)
    modularity_ground = nx.community.modularity(G_ground, communities)
    return modularity_new, modularity_ground

