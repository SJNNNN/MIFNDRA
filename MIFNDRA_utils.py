import dgl
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch as th
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import pairwise_kernels
import cupy as cp
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import pandas as pd
from scipy.sparse import find, csr_matrix
from anndata import AnnData
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, silhouette_score
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return th.LongTensor(edge_index)


def make_adj(edges, size):
    edges_tensor = th.LongTensor(edges).t()
    values = th.ones(len(edges))
    adj = th.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    # adj_dense=adj
    # adj_dense.long()
    return adj


def predict_case(data, args):
    data['m_d_matrix'] = make_adj(data['m_d'], (args.miRNA_number, args.disease_number))
    m_d_matrix = data['m_d_matrix']
    one_index = []
    zero_index = []
    for i in range(m_d_matrix.shape[0]):
        for j in range(m_d_matrix.shape[1]):
            if m_d_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)

    random.shuffle(one_index)
    one_index = np.array(one_index)
    random.shuffle(zero_index)
    zero_index = np.array(zero_index)

    train = np.concatenate(
        (one_index, zero_index[:int(args.negative_rate * len(one_index))]))
    mm = data['mm_f'] * np.where(data['mm_f'] == 0, 0, 1) + get_gaussian(data['m_d_matrix']) * np.where(
        data['mm_f'] == 1, 0, 1)
    dd = data['dd_s'] * np.where(data['dd_s'] == 0, 0, 1) + get_gaussian(data['m_d_matrix'].t()) * np.where(
        data['dd_s'] == 1, 0, 1)
    data['mm'] = {'data_matrix': mm, 'edges': get_edge_index(mm)}
    data['dd'] = {'data_matrix': dd, 'edges': get_edge_index(dd)}
    data['train'] = train




def data_processing(data, args):
    md_matrix = make_adj(data['md'], (args.ncRNA_number, args.drug_number))
    one_index = []
    zero_index = []
    for i in range(md_matrix.shape[0]):
        for j in range(md_matrix.shape[1]):
            if md_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)
    unsamples=[]
    if args.negative_rate == -1:
        zero_index = zero_index
    else:
        unsamples = zero_index[int(args.negative_rate * len(one_index)):]
        zero_index = zero_index[:int(args.negative_rate * len(one_index))]
    index = np.array(one_index + zero_index, int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)

    md = samples[samples[:, 2] == 1, :2]
    md_matrix = make_adj(md, (args.ncRNA_number, args.drug_number))
    md_matrix = md_matrix.numpy()
    # print(md_matrix)
    triplet_samples = []
    ncRNA_drug_map = {}


    for sample in samples:
        if sample[2] == 1:
            ncRNA_idx = sample[0]
            positive_drug_idx = sample[1]
            if ncRNA_idx not in ncRNA_drug_map:
                ncRNA_drug_map[ncRNA_idx] = []
            ncRNA_drug_map[ncRNA_idx].append(positive_drug_idx)


    for ncRNA_idx, positive_drugs in ncRNA_drug_map.items():
        for positive_drug_idx in positive_drugs:

            negative_drug_idx = random.choice([d for d in range(args.drug_number) if d not in positive_drugs])
            triplet_samples.append([ncRNA_idx, positive_drug_idx, negative_drug_idx])

    triplet_samples = np.array(triplet_samples)
    # print(triplet_samples)
    gm = get_gaussian(md_matrix)
    gd = get_gaussian(md_matrix.transpose())
    ms = data['mf'] * data['mfw'] + gm * (1 - data['mfw'])  #
    ds = data['dss'] * data['dsw'] + gd * (1 - data['dsw'])  #


    # ====== 你的原始部分 ======
    m = np.load(args.data_dir + 'SpliceBERTembeddings.npy')
    nc = np.loadtxt(args.data_dir + 'ncRNA_ncRNA_interconnections.txt', dtype=float)
    n = np.load(args.data_dir + 'complete_mol_emb.npy')

    combined_features1 = np.concatenate((ms, m, nc), axis=1)
    n_components = 3322
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=0)
    X_reduced = pca.fit_transform(combined_features1)

    combined_features2 = np.concatenate((ds, n), axis=1)
    n_components = 121
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=0)
    Y_reduced = pca.fit_transform(combined_features2)

    data['ms'] = X_reduced
    data['ds'] = Y_reduced



    data['train_samples'] = samples
    data['train_md'] = md
    data['unsamples']=np.array(unsamples)
    data['md_association'] = md_matrix
    data['triplet_samples'] = triplet_samples

def k_matrix(matrix, k=20):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)





def get_data(args):
    data = dict()

    mf = np.loadtxt(args.data_dir + 'ncRNA_Functional_Similarity_Matrix.txt', dtype=float)#fused_features_final.txt#ncRNA_Functional_Similarity_Matrix.txt# fused_features_final.txt
    mfw = np.loadtxt(args.data_dir + 'ncRNA_Functional_Similarity_weight_Matrix.txt', dtype=float)#weight_matrix_final.txt#ncRNA_Functional_Similarity_weight_Matrix.txt
    ds1 = np.loadtxt(args.data_dir + 'drug_similarity_matrix.txt', dtype=float)

    dsw = np.loadtxt(args.data_dir + 'drug_similarity_weight_Matrix.txt', dtype=float)
    if args.dd2 == True:
        dss = (ds1 + ds2) / 2
    else:
        dss = ds1

    data['ncRNA_number'] = int(mf.shape[0])
    data['drug_number'] = int(dss.shape[0])
    data['mf'] = mf
    data['dss'] = dss
    data['mfw'] = mfw
    data['dsw'] = dsw
    data['d_num'] = 121
    data['m_num'] = 3322
    data['md'] = np.loadtxt(args.data_dir + 'ncrna_rug_index.txt', dtype=int) - 1#ncrna_rug_index.txt#filtered_ncrna_rug_index.txt
    return data



def get_gaussian(adj):
    Gaussian = np.zeros((adj.shape[0], adj.shape[0]), dtype=np.float32)
    gamaa = 1
    sumnorm = 0
    for i in range(adj.shape[0]):
        norm = np.linalg.norm(adj[i]) ** 2
        sumnorm = sumnorm + norm
    gama = gamaa / (sumnorm / adj.shape[0])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            Gaussian[i, j] = math.exp(-gama * (np.linalg.norm(adj[i] - adj[j]) ** 2))

    return Gaussian


