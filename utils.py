###############################################################
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# reference github link : https://github.com/brandeis-machine-learning/FairAdj
#
# @modify : chri0220@skku.edu
# add function fairdropper
#############################################################

import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple, List
import torch


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return


def save(path: str, columns: List[str], data: List[str or float]) -> None:
    if not os.path.isfile(path):
        res = pd.DataFrame(columns=columns)
    else:
        res = pd.read_csv(path)

    curr_res = pd.DataFrame([data], columns=columns)
    res = pd.concat([res, curr_res])
    res.to_csv(path, index=False)

    return

class fairdropper():
    def __init__(self, G, adj, n_epochs, device):
        self.delta = 0.16
        self.device = device
        self.n_epochs = n_epochs
        self.G = G

    def build_drop_map(self, sensitive):
        edges = np.array(list(self.G.edges()))
        src = edges[:, 0]
        dst = edges[:, 1]

        sensitive = torch.LongTensor(sensitive)
        sens_diff = (sensitive[src] != sensitive[dst])
        randomization = torch.FloatTensor(self.n_epochs, sens_diff.size(0)).uniform_() < 0.5 + self.delta

        return sens_diff, randomization

    def drop_fairly(self, epoch, randomization, adj_norm, sens_diff):
        G_new = self.G.copy()
        adj_new = adj_norm.to_dense().clone()

        keep = torch.where(randomization[epoch], sens_diff, ~sens_diff)
        remove_edges = np.array(list(self.G.edges()))[keep]
        remove_edges = [(edge[0], edge[1]) for edge in remove_edges]

        G_new.remove_edges_from(remove_edges)
        for edge in remove_edges:
            adj_new[edge[0], edge[1]] = 0

        return G_new, adj_new.to_sparse().to(self.device)