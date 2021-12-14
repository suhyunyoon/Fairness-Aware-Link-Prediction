###############################################################
# @Author  : Suhyun Yoon, Yonghoon Kang
# @Contact : artfjqm2@skku.edu , evegreen96@skku.edu
#############################################################

import numpy as np
import networkx as nx
import pandas as pd
import os
import csv
import json
from tqdm import tqdm

import pickle as pkl
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple


def get_key(dict_1, value):
    return [k for k, v in dict_1.items() if v == value]

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def build_test(G, nodelist, ratio):
    edges = list(G.edges.data(default=False))
    num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
    num_test = int(np.floor(num_edges * ratio))
    test_edges_true = []
    test_edges_false = []

    # generate false links for testing
    while len(test_edges_false) < num_test:
        idx_u = np.random.randint(0, num_nodes - 1)
        idx_v = np.random.randint(idx_u, num_nodes)

        if idx_u == idx_v:
            continue
        if (idx_u, idx_v) in G.edges(idx_u):
            continue
        if (idx_u, idx_v) in test_edges_false:
            continue
        else:
            test_edges_false.append([idx_u, idx_v, 0])

    # generate true links for testing
    all_edges_idx = list(range(num_edges))
    np.random.shuffle(all_edges_idx)
    test_edges_true_idx = all_edges_idx[:num_test]
    for test_idx in test_edges_true_idx:
        u, v, _ = edges[test_idx]
        G.remove_edge(u, v)
        test_edges_true.append([u, v, 1])

    return G, test_edges_true, test_edges_false

def get_musae_attr(dataname, data_path):
    if dataname == 'LastFM':
        path = os.path.join(data_path, 'LastFM/lasftm_asia/')
        feat = 'lastfm_asia_features.json'
        edge = 'lastfm_asia_edges.csv'
        target = 'lastfm_asia_target.csv'
        s_column = 0 # target
    # elif dataname == 'Facebook':
    #     path = os.path.join(data_path, 'Facebook/')
    #     feat = 'musae_facebook_features.json'
    #     edge = 'musae_facebook_edges.csv'
    #     target = 'musae_facebook_target.csv'
    #     s_column = 2 # pagetype
    elif dataname in ['Twitch_DE', 'Twitch_TW']:
        splitdata = dataname.split('_')
        loc = splitdata[1]
        path = os.path.join(data_path, f'Twitch/{loc}/')
        feat = f'musae_{loc}.json'
        edge = f'musae_{loc}_edges.csv'
        target = f'musae_{loc}_target.csv'
        s_column = 3 #1
    elif dataname in ['Twitch_ENGB', 'Twitch_ES', 'Twitch_FR', 'Twitch_PTBR', 'Twitch_RU']:
        splitdata = dataname.split('_')
        loc = splitdata[1]
        path = os.path.join(data_path, f'Twitch/{loc}/')
        feat = f'musae_{loc}_features.json'
        edge = f'musae_{loc}_edges.csv'
        target = f'musae_{loc}_target.csv'
        s_column = 3 #1
    elif dataname == 'Github':
        path = os.path.join(data_path, 'Github/git_web_ml/')
        feat = 'musae_git_features.json'
        edge = 'musae_git_edges.csv'
        target = 'musae_git_target.csv'
        s_column = 1
    else:
        print('Not supported dataset')
        path = None
        feat = None
        edge = None
        target = None
        s_column = None
    return path, feat, edge, target, s_column

def load_musae_dataset(dataname, data_path):
    print(f'Loading {dataname} dataset')
    path, feat, edge, target, s_column = get_musae_attr(dataname, data_path=data_path)
    if path != None:
        edge = open(path+edge, "r", encoding='utf8')
        next(edge, None)  # skip the first line in the input file
        with open(path+target, mode='r') as inp:
            reader = csv.reader(inp)
            next(reader, None)
            target = {rows[0]:rows[1:] for rows in reader}
        #print(target)
        Graphtype = nx.Graph()
        G = nx.parse_edgelist(edge, delimiter=',', create_using=Graphtype,
                              nodetype=int, data=(('weight', float),))

        nodelist = {idx: node for idx, node in enumerate(target.keys())}
        #print(nodelist)

        if os.path.isfile(path+'feature.npy'):
            X = np.load(path+'feature.npy')
        else:
            with open(path+feat) as json_file:
                features = json.load(json_file)
            X = list()
            keys = features.keys()
            dic_max = 0
            for max_val in features.values():
                if len(max_val) == 0:
                    continue
                if max(max_val) > dic_max:
                    dic_max = max(max_val)
            real_max = dic_max + 1
            for key in tqdm(keys):
                value = features[key]
                feature = list()
                for i in range(real_max):
                    if i in value:
                        feature.append(1)
                    else:
                        feature.append(0)
                X.append(feature)
            X = np.array(X)
            np.save(path+'feature.npy', X)

        sensitive = [v[s_column] for k, v in target.items()]
        G, test_edges_true, test_edges_false = build_test(G, nodelist, 0.1)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
        return G, adj, X, sensitive, test_edges_true, test_edges_false, nodelist
    else:
        print('load failed')

def get_facebook_features(ego_nodes, dataset_directory):
    if dataset_directory[-1] != '/':
        dataset_directory = dataset_directory + '/'

    features = {}
    gender_featnum = []
    for ego_node in ego_nodes:
        featnames = open(dataset_directory + str(ego_node) + '.featnames')
        features_node = []
        for line in featnames:
            features_node.append(int(line.split(';')[-1].split()[-1]))
            if line.split(';')[0].split()[1] == 'gender':
                if int(line.split(';')[-1].split()[-1]) not in gender_featnum:
                    gender_featnum.append(int(line.split(';')[-1].split()[-1]))
        features[ego_node] = features_node
        featnames.close()
    gender_featnum.sort()
    return features, gender_featnum

def load_facebook_dataset(data_path):
    # Read edges & construct whole graph
    edge = os.path.join(data_path, 'facebook_combined.txt')
    edges = []
    with open(edge) as f:
        print('file opened')
        for i, line in enumerate(f):
            words = line.split()
            edges.append((int(words[0]), int(words[1])))
        print('Reading edges finished')

    G = nx.Graph(edges)
    nodes = ['0', '107', '348', '414', '686', '698', '1684', '1912', '3437', '3980']
    # get gender feature index
    features_idx, gender_featnum = get_facebook_features(nodes, os.path.join(data_path, 'facebook'))
    # numpy features, # Nodes * # Features
    X = np.zeros((len(G.nodes()), max(map(max, features_idx.values()))+1))
    # read all features(targets)
    target = dict()
    for n in nodes:
        gender_idx = [features_idx[n].index(gender_featnum[0]),
                      features_idx[n].index(gender_featnum[1])]
        # read all targets
        feat = os.path.join(data_path, 'facebook', f'{n}.feat')
        with open(feat) as f:
            for i, line in enumerate(f):
                feats = line.split()
                target[int(feats[0])] = feats[1:]
                X[int(feats[0])][features_idx[n]] = np.array(feats[1:], dtype=np.float64)
        # read ego targets
        egofeat = os.path.join(data_path, 'facebook', f'{n}.egofeat')
        with open(egofeat) as f:
            for i, line in enumerate(f):
                feats = line.split()
                target[int(n)] = feats
                X[int(n)][features_idx[n]] = np.array(feats, dtype=np.float64)
    # sensitive
    sensitive = X[:, gender_featnum[-1]]
    communities = [set(np.where(sens == sensitive)[0].tolist()) for sens in np.unique(sensitive)]
    # nodelist and data split
    nodelist = {idx: node for idx, node in enumerate(target.keys())}

    # updated sensitive by splitted G
    G, test_edges_true, test_edges_false = build_test(G, nodelist, 0.1)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    # adj matrix
    adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))

    return G, adj, X, sensitive, communities, test_edges_true, test_edges_false, nodelist

cora_label = {
        "Genetic_Algorithms": 0,
        "Reinforcement_Learning": 1,
        "Neural_Networks": 2,
        "Rule_Learning": 3,
        "Case_Based": 4,
        "Theory": 5,
        "Probabilistic_Methods": 6,
}

def load_cora_dataset(data_path, scale=True,
         test_ratio=0.1) -> Tuple:
    feat_path = os.path.join(data_path, "cora.content")
    edge_path = os.path.join(data_path, "cora.cites")

    idx_features_labels = np.genfromtxt(feat_path, dtype=np.dtype(str))
    idx_features_labels = idx_features_labels[idx_features_labels[:, 0].astype(np.int32).argsort()]

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    nodelist = {idx: node for idx, node in enumerate(idx)}
    X = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    sensitive = np.array(list(map(cora_label.get, idx_features_labels[:, -1])))

    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    G = nx.read_edgelist(edge_path, nodetype=int)
    G = nx.convert_node_labels_to_integers(G)

    G, test_edges_true, test_edges_false = build_test(G, nodelist, test_ratio)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))

    communities = [set(np.where(sens == sensitive)[0].tolist()) for sens in np.unique(sensitive)]

    return G, adj, X, sensitive, communities, test_edges_true, test_edges_false, nodelist


def load_citeseer_dataset(data_dir, scale=True, test_ratio=0.1) -> Tuple:
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open(os.path.join(data_dir, "ind.citeseer.{}".format(names[i])), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file(os.path.join(data_dir, "ind.citeseer.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range - min(test_idx_range), :] = ty
    ty = ty_extended

    X = sp.vstack((allx, tx)).tolil()
    X[test_idx_reorder, :] = X[test_idx_range, :]
    X = X.toarray()
    onehot_labels = np.vstack((ally, ty))
    onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]

    sensitive = np.argmax(onehot_labels, 1)
    communities = [set(np.where(sens == sensitive)[0].tolist()) for sens in np.unique(sensitive)]

    G = nx.from_dict_of_lists(graph)
    G = nx.convert_node_labels_to_integers(G)

    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    nodes = sorted(G.nodes())
    nodelist = {idx: node for idx, node in zip(range(G.number_of_nodes()), list(nodes))}

    G, test_edges_true, test_edges_false = build_test(G, nodelist, test_ratio)

    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    adj = nx.adjacency_matrix(G, nodelist=nodes)

    return G, adj, X, sensitive, communities, test_edges_true, test_edges_false, nodelist

# Directory path which contains dataset directory(ex. data_path/Facebook/)
def load_dataset(dataname, data_path='./dataset'):
    if dataname == 'Facebook':
        return load_facebook_dataset(os.path.join(data_path, 'Facebook'))
    elif dataname == 'cora':
        return load_cora_dataset(os.path.join(data_path, 'cora'))
    elif dataname == 'citeseer':
        return load_citeseer_dataset(os.path.join(data_path, 'citeseer'))
    else:
        # musae datasets
        return load_musae_dataset(dataname, data_path)

### USAGE ###
if __name__ == '__main__':
    print('Loading Facebook...')
    G, adj, X, sensitive, test_edges_true, test_edges_false, nodelist = load_dataset('Facebook', data_path='data')
    with open('facebook/G.pickle', 'wb') as f:
        pkl.dump(G, f, pkl.HIGHEST_PROTOCOL)
    with open('facebook/X.pickle', 'wb') as f:
        pkl.dump(X, f, pkl.HIGHEST_PROTOCOL)
    with open('facebook/sensitive.pickle', 'wb') as f:
        pkl.dump(sensitive, f, pkl.HIGHEST_PROTOCOL)
    with open('facebook/nodelist.pickle', 'wb') as f:
        pkl.dump(nodelist, f, pkl.HIGHEST_PROTOCOL)
    print(len(G.nodes()), len(G.edges))
    print(len(sensitive), len(nodelist))


    # print('Loading Cora...')
    # G, adj, X, sensitive, test_edges_true, test_edges_false, nodelist = load_dataset('cora', data_path='./data')
    # print(len(G.nodes()), len(G.edges))
    # print(len(sensitive), len(nodelist))
    # print('Loading Citeseer...')
    # G, adj, X, sensitive, test_edges_true, test_edges_false, nodelist = load_dataset('citeseer', data_path='./data')
    # print(len(G.nodes()), len(G.edges))
    # print(len(sensitive), len(nodelist))
