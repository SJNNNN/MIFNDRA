# from MAMFGAT import MAMFGAT
import copy

# from utils import mutual_information_graph
import dgl
import networkx as nx
import torch as th
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve
from sklearn.model_selection import KFold
from torch import optim, nn
from tqdm import trange

from MIFNDRA import MIFNDRA
from MIFNDRA_utils import k_matrix
from sklearn.metrics import brier_score_loss
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
import pandas as pd



def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'precision ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'f1_score ：%.4f \n' % (list[5]))


def print_met2(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'precision ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'f1_score ：%.4f \n' % (list[5]))

def train(data, args):
    import numpy as np
    kfolds = 5
    all_score = []
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
    train_idx, valid_idx = [], []
    for train_index, valid_index in kf.split(data['train_samples']):
        train_idx.append(train_index)
        valid_idx.append(valid_index)
    max_test_auc = 0
    tpr_list = []
    fpr_list = []
    recall_list = []
    precision_list = []
    roc_auc_list = []
    auprc_list = []
    for j in range(kfolds):
        one_score = []
        model = MIFNDRA(args).to(device)
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
        cross_entropy = nn.BCELoss()
        triplet_loss_fn = nn.TripletMarginLoss(margin=0.5, p=2)
        # triplet_loss_fn = AdaptiveTripletMarginLoss(initial_margin=0.5, p=2.0, margin_update_rate=0.0001)

        ncRNA = data['ms']
        drug = data['ds']
        a, b = data['train_samples'][train_idx[j]], data['train_samples'][valid_idx[j]]
        c = data['triplet_samples']
        print(f'################Fold {j + 1} of {kfolds}################')
        epochs = trange(args.epochs, desc='train')
        for _ in epochs:
            alpha = nn.Parameter(th.tensor(1.0), requires_grad=True)
            beta = nn.Parameter(th.tensor(1.0), requires_grad=True)
            model.train()

            optimizer.zero_grad()

            mm_matrix = k_matrix(data['ms'], args.neighbor)
            dd_matrix = k_matrix(data['ds'], args.neighbor)
            # visualize_graph(mm_matrix)
            mm_nx = nx.from_numpy_array(mm_matrix)
            dd_nx = nx.from_numpy_array(dd_matrix)
            mm_graph = dgl.from_networkx(mm_nx)
            dd_graph = dgl.from_networkx(dd_nx)
            md_copy = copy.deepcopy(data['train_md'])

            md_copy[:, 1] = md_copy[:, 1] + args.ncRNA_number
            md_graph = dgl.graph(
                (np.concatenate((md_copy[:, 0], md_copy[:, 1])), np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
                num_nodes=args.ncRNA_number + args.drug_number)
            ncRNA_th = th.Tensor(ncRNA)
            drug_th = th.Tensor(drug)
            # train_samples_th = th.Tensor(data['train_samples']).float()
            train_samples_th = th.Tensor(a).float()
            train_score, anchor_mm, positive_dd, negative_dd = model(mm_graph, dd_graph, md_graph, ncRNA_th, drug_th,
                                                                     a, c)

            # Compute triplet loss
            triplet_loss = triplet_loss_fn(anchor_mm, positive_dd, negative_dd)
            # print(train_samples_th[:, 2])
            # print(th.flatten(train_score))
            # loss_mutual = loss_mutual_information(d1, d2, 64, 16) + loss_mutual_information(m1, m2, 64, 16)
            train_cross_loss = cross_entropy(th.flatten(train_score), train_samples_th[:, 2].to(device))

            train_loss = alpha * train_cross_loss + beta * triplet_loss
            # train_loss = train_cross_loss
            scoree, _, _, _ = model(mm_graph, dd_graph, md_graph, ncRNA_th, drug_th, b, c)
            scoree = scoree.cpu()
            scoree = scoree.detach().numpy()
            # score=score.detach().numpy()

            sc = data['train_samples'][valid_idx[j]]
            sc_true = sc[:, 2]
            aucc = roc_auc_score(sc_true, scoree)

            print("AUC=", np.round(aucc, 4), "l_1=", np.round(triplet_loss.item(), 4), "loss=",
                  np.round(train_loss.item(), 4))
            train_loss.backward()
            # print(train_loss.item())
            optimizer.step()

        model.eval()


        scoree, _, _, _ = model(mm_graph, dd_graph, md_graph, ncRNA_th, drug_th, b, c)

        scoree = scoree.cpu()
        scoree = scoree.detach().numpy()

        import numpy as np


        emb_mm = positive_dd.detach().cpu().numpy()
        emb_dd = negative_dd.detach().cpu().numpy()


        embeddings = np.concatenate((emb_mm, emb_dd), axis=0)


        sc = data['train_samples'][valid_idx[j]]
        sc_true = sc[:, 2]

        labels = np.concatenate(([1] * len(emb_mm), [0] * len(emb_dd)), axis=0)

        sc = data['train_samples'][valid_idx[j]]
        sc_true = sc[:, 2]

        fold_results = pd.DataFrame({
            'True Labels': sc_true,
            'Predictions': scoree.ravel()
        })
        # fold_results.to_csv(f'result/ncRNA_Drug/1:10/fold_{j + 1}_results.csv', index=False)  # Saving to CSV

        fpr, tpr, thresholds = roc_curve(sc_true, scoree)
        # 选择最佳阈值
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print("Best threshold：{:.4f}".format(optimal_threshold))
        fpr, tpr, _ = roc_curve(sc_true, scoree)
        roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, lw=3, label=f'{j + 1}Fold ROC curve (area = {roc_auc:.4f})')

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)

        aucc = roc_auc_score(sc_true, scoree)
        # if aucc > max_test_auc:
        #     th.save(model.state_dict(), "./save_model/5_fold/HMDD v2.0_5fold_train_model.pth")
        # if aucc > max_test_auc:
        #     th.save(model.state_dict(), "./save_model/ncRNA_drug/best_model.pth")
        precision, recall, thresholds = precision_recall_curve(sc_true, scoree)
        roc_pr = auc(recall, precision)
        # plt.plot(recall, precision, lw=2, label=f'{j + 1}Fold PR curve (area = { roc_pr:.4f})')
        recall_list.append(recall)
        precision_list.append(precision)
        print("AUC: {:.6f}".format(aucc))

        auprc = auc(recall, precision)
        auprc_list.append( auprc)
        print("AUPRC: {:.6f}".format(auprc))

        # ===== 再基于最佳阈值转成0/1用于分类指标 =====
        scoree_binary = scoree.copy()
        for i in range(len(scoree_binary)):
            if scoree_binary[i] >= optimal_threshold:
                scoree_binary[i] = 1
            else:
                scoree_binary[i] = 0

        scoree = np.array(scoree)
        scoree = scoree.ravel()

        for i in range(len(scoree)):
            if scoree[i] >= optimal_threshold:
                scoree[i] = 1
            else:
                scoree[i] = 0
        accuracy = accuracy_score(sc_true, scoree)
        print("Accuracy: {:.6f}".format(accuracy))
        precision = precision_score(sc_true, scoree)
        print("Precision: {:.6f}".format(precision))
        recall = recall_score(sc_true, scoree)
        print("Recall: {:.6f}".format(recall))
        f1 = f1_score(sc_true, scoree)
        print("F1-score: {:.6f}".format(f1))
        one_score = [aucc, auprc,  accuracy, precision, recall, f1]
        all_score.append(one_score)
    cv_metric = np.mean(all_score, axis=0)
    sD_metric = np.std(all_score, axis=0)
    print('################5-Fold Result################')
    print_met(cv_metric)
    print_met2(sD_metric)

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    import numpy as np

    # —— 在这里准备你的数据 —— #
    # fpr_list: list of arrays, 每个 array 是该折的 False Positive Rate
    # tpr_list: list of arrays, 每个 array 是该折的 True Positive Rate
    # roc_auc_list: list of floats, 每折 ROC AUC
    # recall_list: list of arrays, 每个 array 是该折的 Recall
    # precision_list: list of arrays, 每个 array 是该折的 Precision
    # auprc_list: list of floats, 每折 PR AUC
    # cv_metric: tuple or list, (mean_roc_auc, mean_auprc)
    # kfolds: int, 折数

    # 示例（请用实际结果替换）：
    # fpr_list = [...]
    # tpr_list = [...]
    # roc_auc_list = [...]
    # recall_list = [...]
    # precision_list = [...]
    # auprc_list = [...]
    # cv_metric = (0.9123, 0.8456)
    # kfolds = 5

    # ------------------------------------------------ #

    # 创建画布和子图
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # ----------------- ROC 曲线 ----------------- #
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(
        [np.interp(mean_fpr, fpr_list[i], tpr_list[i]) for i in range(kfolds)],
        axis=0
    )

    colors = ['b', 'g', 'r', 'c', 'm']  # 根据 kfolds 数量选色

    # 每一折
    for i in range(kfolds):
        ax1.plot(
            fpr_list[i], tpr_list[i],
            lw=2, color=colors[i],
            label=f'Fold {i + 1} ROC (AUC={roc_auc_list[i]:.4f})'
        )

    # 平均 ROC
    ax1.plot(
        mean_fpr, mean_tpr,
        color='k', linestyle='--', lw=1,
        label=f'Mean ROC (AUC={cv_metric[0]:.4f})'
    )

    # 放大坐标轴字体
    ax1.set_xlabel('False Positive Rate', fontsize=16)
    ax1.set_ylabel('True Positive Rate', fontsize=16)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc='lower right')

    # ROC 局部放大图
    axins = inset_axes(ax1, width="35%", height="30%", loc='center right', borderpad=3)
    for i in range(kfolds):
        axins.plot(fpr_list[i], tpr_list[i], lw=1, color=colors[i])
    axins.plot(mean_fpr, mean_tpr, color='k', linestyle='--', lw=1)

    # 放大区域范围
    x1, x2, y1, y2 = 0.0, 0.3, 0.8, 1.0
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.tick_params(axis='both', labelsize=12)

    # 标记放大区域
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         edgecolor='black', linestyle='--', facecolor='none', linewidth=1)
    ax1.add_patch(rect)
    mark_inset(ax1, axins, loc1=3, loc2=1, fc="none", ec="black",
               linestyle="--", linewidth=1)
    for spine in axins.spines.values():
        spine.set_linestyle('--')
        spine.set_linewidth(1)

    # ----------------- PR 曲线 ----------------- #
    mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)
    for i in range(kfolds):
        # 注意对 recall-precision 反向插值
        pr_interp = np.interp(mean_recall,
                              recall_list[i][::-1],
                              precision_list[i][::-1])
        mean_precision += pr_interp
    mean_precision /= kfolds

    # 每一折
    for i in range(kfolds):
        ax2.plot(
            recall_list[i], precision_list[i],
            lw=1, color=colors[i],
            label=f'Fold {i + 1} PR (AUC={auprc_list[i]:.4f})'
        )

    # 平均 PR
    ax2.plot(
        mean_recall, mean_precision,
        color='k', linestyle='--', lw=2,
        label=f'Mean PR (AUC={cv_metric[1]:.4f})'
    )

    # 放大坐标轴字体
    ax2.set_xlabel('Recall', fontsize=16)
    ax2.set_ylabel('Precision', fontsize=16)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc='lower left')

    # PR 局部放大图
    axins2 = inset_axes(ax2, width="35%", height="30%", loc='center left', borderpad=3)
    for i in range(kfolds):
        axins2.plot(recall_list[i], precision_list[i], lw=1, color=colors[i])
    axins2.plot(mean_recall, mean_precision, color='k', linestyle='--', lw=1)

    x1, x2, y1, y2 = 0.7, 1.0, 0.7, 1.0
    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1, y2)
    axins2.tick_params(axis='both', labelsize=12)

    rect2 = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          edgecolor='black', linestyle='--', facecolor='none', linewidth=1)
    ax2.add_patch(rect2)
    mark_inset(ax2, axins2, loc1=2, loc2=4, fc="none", ec="black",
               linestyle="--", linewidth=1)

    # # 保存图片
    # output_directory = './result/ncRNA_Drug/'
    # output_filename = 'fivefold_ROC_PR_curve_test2.png'
    # plt.savefig(f'{output_directory}/{output_filename}', dpi=300, bbox_inches='tight')

    plt.close()

