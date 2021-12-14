###############################################################
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# reference github link : https://github.com/brandeis-machine-learning/FairAdj
#
# @modify : chri0220@skku.edu, maya0707@skku.edu
# add function
# 1. dataloader
# 2. fairdrop
# 3. greedy
# 4. evaluation
#############################################################

import numpy as np
from typing import Sequence, Tuple, List
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx

THRE = 0.5

def fair_link_eval(
        G, nodelist, communities,
        emb: np.ndarray,
        sensitive: np.ndarray,
        test_edges_true: Sequence[Tuple[int, int]],
        test_edges_false: Sequence[Tuple[int, int]],
        dataset=None,
        rec_ratio: List[float] = None,

) -> Sequence[List]:

    cora = False

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.array(np.dot(emb, emb.T), dtype=np.float128)

    print( 'adj_rec : ' , len(adj_rec)) 

    ## calculate modularity
    if G.is_directed():
        G = G.to_undirected()
    G_ground = G.copy()
    G_new = G.copy()

    preds_pos_intra = []
    preds_pos_inter = []
    for e in test_edges_true:
        ## to calculate modularity
        # for cora, citeseer
        if cora:
            G_ground.add_edge(nodelist[e[0]], nodelist[e[1]])
            if adj_rec[e[0], e[1]] > THRE:
                G_new.add_edge(nodelist[e[0]], nodelist[e[1]])
        # for twitch
        else:
            G_ground.add_edge(e[0], e[1])
            if adj_rec[e[0], e[1]] > THRE:
                G_new.add_edge(e[0], e[1])

        if sensitive[e[0]] == sensitive[e[1]]:
            preds_pos_intra.append(sigmoid(adj_rec[e[0], e[1]]))
        else:
            preds_pos_inter.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_neg_intra = []
    preds_neg_inter = []
    for e in test_edges_false:
        ## to calculate modularity
        if cora:
            # for cora, citeseer
            if adj_rec[e[0], e[1]] > THRE:
                G_new.add_edge(nodelist[e[0]], nodelist[e[1]])

        # for twitch
        else:
            if adj_rec[e[0], e[1]] > THRE:
                G_new.add_edge(e[0], e[1])

        if sensitive[e[0]] == sensitive[e[1]]:
            preds_neg_intra.append(sigmoid(adj_rec[e[0], e[1]]))
        else:
            preds_neg_inter.append(sigmoid(adj_rec[e[0], e[1]]))

    res = {}
    for preds_pos, preds_neg, type in zip((preds_pos_intra, preds_pos_inter, preds_pos_intra + preds_pos_inter),
                                          (preds_neg_intra, preds_neg_inter, preds_neg_intra + preds_neg_inter),
                                          ("intra", "inter", "overall")):
        preds_all = np.hstack([preds_pos, preds_neg])
        labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        err = (np.sum(list(map(lambda x: x >= THRE, preds_pos))) + np.sum(
            list(map(lambda x: x < THRE, preds_neg)))) / (len(preds_pos) + len(preds_neg))

        score_avg = (sum(preds_pos) + sum(preds_neg)) / (len(preds_pos) + len(preds_neg))
        pos_avg, neg_avg = sum(preds_pos) / len(preds_pos), sum(preds_neg) / len(preds_neg)

        res[type] = [roc_score, ap_score, err, score_avg, pos_avg, neg_avg]


    ks_pos = stats.ks_2samp(preds_pos_intra, preds_pos_inter)[0]
    ks_neg = stats.ks_2samp(preds_neg_intra, preds_neg_inter)[0]

    # calculate modularity
    modularity_new = nx.community.modularity(G_new, communities)
    modularity_ground = nx.community.modularity(G_ground, communities)
    modred = (modularity_ground - modularity_new) / np.abs(modularity_ground)

    standard = res["overall"][0:2] + [modred] + [abs(res["intra"][i] - res["inter"][i]) for i in range(3, 6)] + [ks_pos, ks_neg]

    return standard
