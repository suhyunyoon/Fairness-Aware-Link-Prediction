###############################################################
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# reference github link : https://github.com/brandeis-machine-learning/FairAdj
#
# @modify : artfjqm2@skku.edu, evegreen96@skku.edu, maya0707@skku.edu
# add function 
# 1. dataloader 
# 2. fairdrop
# 3. greedy
# 4. evaluation 
#############################################################

import scipy.sparse as sp

import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

from args import parse_args
from utils import fix_seed, fairdropper
from get_data_v2 import load_dataset
from model.utils import preprocess_graph, project
from model.optimizer import loss_function
from model.gae import GCNModelVAE
from model.greedy import greedy_wrapper
from greedy_eval import get_accuracy_scores, get_modularity, get_fairness_scores
from eval import fair_link_eval



import os 

def main(args):
    # Data preparation
    G, adj, features, sensitive, communities, test_edges_true, test_edges_false, nodelist = load_dataset(args.dataset, 'data')

    n_nodes, feat_dim = features.shape
    features = torch.from_numpy(features).float().to(args.device)
    sensitive_save = sensitive.copy()

    adj_norm = preprocess_graph(adj).to(args.device)
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    adj_label = torch.FloatTensor(adj.toarray()).to(args.device)

    # drop edges fairly
    fairdrop = fairdropper(G, adj, args.n_epochs, args.device)
    sens_diff, randomization = fairdrop.build_drop_map(sensitive)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    pos_weight = torch.Tensor([pos_weight]).to(args.device)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout) #.to(args.device)
    optimizer = optim.Adam(model.get_parameters(), lr=args.lr)
    
    model = model.to(args.device)

    print( 'start train -------------------------------------')
    # Training
    model.train()
    for epoch in range(args.n_epochs):
        # print("-"*100)
        # print(f"Epoch {epoch}")

        if epoch == 0 or epoch % args.fairdrop == 0:
            _, adj_new = fairdrop.drop_fairly(epoch, randomization, adj_norm, sens_diff)

        optimizer.zero_grad()
        recovered, z, mu, logvar = model(features, adj_new)
        loss = loss_function(preds=recovered, labels=adj_label, mu=mu, logvar=logvar,
                             n_nodes=n_nodes, norm=norm, pos_weight=pos_weight)

        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        print("Epoch: [{:d}];".format(epoch + 1), "Loss: {:.3f};".format(cur_loss))


    print( 'start eval ------------------------------------')
    # Evaluation
    model.eval()
    with torch.no_grad():
        output = model(features, adj_norm)
        print("output [0] original : ", output[0].shape)
        print("output [1] : ", output[1].shape)
        z = output[0]	

    # 2708 * 2708 
    hidden_emb = z.data.cpu().numpy()

    print('args.greedy ',  args.greedy)
    if args.greedy:
        thresh = np.median(hidden_emb)
        # [source node , destination node , 1 or 0 ]
        #combine test_edges_true + test_edges_false 
        test_edges = test_edges_true + test_edges_false
 
        pred_temp = [hidden_emb[i, j] for i , j, _ in test_edges]
        print( 'pred_temp.shape ' , len(pred_temp) )
    
        new_pred = greedy_wrapper(G, communities, test_edges, pred_temp, thresh, args.greedy_change_pct)

        # calculate accurcay and modred of postprocessing
        _, roc_auc, ap_score = get_accuracy_scores(np.array(test_edges), new_pred)
        fairness_score = get_fairness_scores(G, communities, np.array(test_edges), new_pred)
        modularity_new, modularity_ground = get_modularity(G, new_pred, communities, np.array(test_edges))
        modred = (modularity_ground - modularity_new) / np.abs(modularity_ground)

        scores = [roc_auc, ap_score, modred] + fairness_score
        print("Result : Greedy postprocessing ------")
        print('METRIC\t' + ' '.join(
            '{0:>8}'.format(metric) for metric in ["auc", "ap", "modred", "dp", "true", "false", "fnr", "tnr"]))
        print('VALUE\t' + ' '.join('{0:>8.4f}'.format(value) for value in scores))

    else:
        # evaluation
        std = fair_link_eval(G, nodelist, communities, hidden_emb, sensitive_save, test_edges_true, test_edges_false, args.dataset)
        print("Result below ------")
        print('METRIC\t' + ' '.join('{0:>8}'.format(metric) for metric in ["auc", "ap", "modred", "dp", "true", "false", "fnr", "tnr"]))
        print('VALUE\t'  + ' '.join('{0:>8.4f}'.format(value) for value in std))

        # for term, val in zip(col, std):
        #    print("{0:>8}: {1:>8.4f}".format(term, val))


    return


if __name__ == "__main__":
    args = parse_args()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = " 2,3"

    args.device = torch.device(args.device)
    fix_seed(args.seed)
    main(args)
