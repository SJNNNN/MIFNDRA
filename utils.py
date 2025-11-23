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
    md_matrix = make_adj(data['md'], (args.miRNA_number, args.disease_number))
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
    # print(samples.shape)
 
    md = samples[samples[:, 2] == 1, :2]
    md_matrix = make_adj(md, (args.miRNA_number, args.disease_number))
    md_matrix = md_matrix.numpy()
    # print(md_matrix)
    triplet_samples = []
    miRNA_disease_map = {}  


    for sample in samples:
        if sample[2] == 1:  
            miRNA_idx = sample[0]
            positive_disease_idx = sample[1]
            if miRNA_idx not in miRNA_disease_map:
                miRNA_disease_map[miRNA_idx] = []
            miRNA_disease_map[miRNA_idx].append(positive_disease_idx)

    # 生成三元组
    for miRNA_idx, positive_diseases in miRNA_disease_map.items():
        for positive_disease_idx in positive_diseases:
      
            negative_disease_idx = random.choice([d for d in range(args.disease_number) if d not in positive_diseases])
            triplet_samples.append([miRNA_idx, positive_disease_idx, negative_disease_idx])

    triplet_samples = np.array(triplet_samples)
    # print(triplet_samples)
    gm = get_gaussian(md_matrix)
    gd = get_gaussian(md_matrix.transpose())
    ms = data['mf'] * data['mfw'] + gm * (1 - data['mfw'])  #
    ds = data['dss'] * data['dsw'] + gd * (1 - data['dsw'])  #



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
    features = samples




    def corr_mx(Z):
        C = np.corrcoef(Z, rowvar=False)
        return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    def plot_corr(C, title, filename, k=50):
        plt.figure(figsize=(6, 5))
        im = plt.imshow(C[:k, :k], vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        # plt.title(f"{title} (first {k} dims)", fontsize=14)
        plt.xlabel("feature index", fontsize=14)
        plt.ylabel("feature index", fontsize=14)
        plt.tight_layout()

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
     
    #
    def safe_plot(X_raw, X_pca, pca, prefix, k=50):
        print(f"== {prefix} ==")
        print("X_reduced shape:", X_pca.shape)
        print("any NaN?", np.isnan(X_pca).any())
        print("explained variance (前10):", pca.explained_variance_[:10])

        # 只保留方差 > eps 的PC
        eps = 1e-12
        valid_idx = np.where(pca.explained_variance_ > eps)[0]
        if valid_idx.size == 0:
            print(f"[{prefix}] 所有PC方差接近0，无法画PCA热图")
            return
        k_eff = min(k, valid_idx.size)


        C_raw = corr_mx(X_raw[:, :k_eff])
        plot_corr(C_raw, f"{prefix} Raw correlation", filename=f"result/PCA/{prefix}_Raw_corr.png", k=k_eff)


        C_pca = corr_mx(X_pca[:, valid_idx[:k_eff]])
        plot_corr(C_pca, f"{prefix} PCA correlation", filename=f"result/PCA/{prefix}_PCA_corr.png", k=k_eff)

    safe_plot(combined_features1, X_reduced, pca, "ncRNA", k=50)
    safe_plot(combined_features2, Y_reduced, pca, "Drug", k=50)
    #
    def offdiag_stats(C):
        n = C.shape[0]
        mask = ~np.eye(n, dtype=bool)
        vals = np.abs(C[mask])
        return {
            "mean_abs_corr": float(vals.mean()),
            "median_abs_corr": float(np.median(vals)),
            "p95_abs_corr": float(np.percentile(vals, 95)),
            "frac_abs_corr_ge_0.5": float((vals >= 0.5).mean()),
            "max_abs_corr": float(vals.max())
        }

    def safe_plot(X_raw, X_pca, pca, prefix, k=50):
        print(f"== {prefix} ==")
        print("X_reduced shape:", X_pca.shape)
        print("any NaN?", np.isnan(X_pca).any())
        print("explained variance (前10):", pca.explained_variance_[:10])

        eps = 1e-12
        valid_idx = np.where(pca.explained_variance_ > eps)[0]
        if valid_idx.size == 0:
            print(f"[{prefix}] 所有PC方差接近0，无法画PCA热图")
            return
        k_eff = min(k, valid_idx.size)


        C_raw = corr_mx(X_raw[:, :k_eff])
        C_pca = corr_mx(X_pca[:, valid_idx[:k_eff]])


        # plot_corr(C_raw, f"{prefix} Raw correlation", k_eff)
        # plot_corr(C_pca, f"{prefix} PCA correlation", k_eff)


        stats_raw = offdiag_stats(C_raw)
        stats_pca = offdiag_stats(C_pca)

        print(f"[{prefix}] Off-diagonal correlation metrics (first {k_eff} dims)")
        print("  RAW :",
              f"mean={stats_raw['mean_abs_corr']:.4f},",
              f"median={stats_raw['median_abs_corr']:.4f},",
              f"p95={stats_raw['p95_abs_corr']:.4f},",
              f"frac(|corr|≥0.5)={stats_raw['frac_abs_corr_ge_0.5']:.4f},",
              f"max={stats_raw['max_abs_corr']:.4f}")
        print("  PCA :",
              f"mean={stats_pca['mean_abs_corr']:.4e},",  # 用科学计数法看更清楚
              f"median={stats_pca['median_abs_corr']:.4e},",
              f"p95={stats_pca['p95_abs_corr']:.4e},",
              f"frac(|corr|≥0.5)={stats_pca['frac_abs_corr_ge_0.5']:.4f},",
              f"max={stats_pca['max_abs_corr']:.4e}")

    safe_plot(combined_features1, X_reduced, pca, "ncRNA", k=50)
    safe_plot(combined_features2, Y_reduced, pca, "Drug", k=50)

    #
    #
    def plot_cumvar_curves(X_list, labels, prefix="PCA"):
        plt.figure(figsize=(7, 5))

        for X, label in zip(X_list, labels):
            # 完整 PCA
            pca_full = PCA(n_components=min(X.shape), svd_solver="randomized", random_state=0)
            pca_full.fit(X)

            # 累计方差
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)

            # 绘制曲线
            plt.plot(np.arange(1, len(cumvar) + 1), cumvar, label=label)

        plt.axhline(0.95, color="gray", linestyle="--", label="95% variance")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.title(f"{prefix} Cumulative Variance Curves")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ===== 调用 =====
    plot_cumvar_curves(
        [combined_features1, combined_features2],
        ["ncRNA features", "Drug features"],
        prefix="ncRNA & Drug"
    )



    # 绘制t-SNE图


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

    data['miRNA_number'] = int(mf.shape[0])
    data['disease_number'] = int(dss.shape[0])
    data['mf'] = mf
    data['dss'] = dss
    data['mfw'] = mfw
    data['dsw'] = dsw
    data['d_num'] = 121
    data['m_num'] = 3322
    data['md'] = np.loadtxt(args.data_dir + 'ncrna_drug_index.txt', dtype=int) - 1#ncrna_rug_index.txt#filtered_ncrna_rug_index.txt
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




