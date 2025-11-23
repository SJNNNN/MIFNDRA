# from MAMFGAT import MAMFGAT
from torch import optim, nn
from tqdm import trange
from MIFNDRA_utils import k_matrix
from MIFNDRA import MIFNDRA
from sklearn.manifold import TSNE
from scipy import interp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import ConnectionPatch
# from utils import mutual_information_graph
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import networkx as nx
import numpy as np
import copy
import torch as th
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve
from sklearn.model_selection import KFold
import torch.nn.functional as F
import scipy.sparse as sp
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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

        miRNA = data['ms']
        disease = data['ds']
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

            md_copy[:, 1] = md_copy[:, 1] + args.miRNA_number
            md_graph = dgl.graph(
                (np.concatenate((md_copy[:, 0], md_copy[:, 1])), np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
                num_nodes=args.miRNA_number + args.disease_number)
            miRNA_th = th.Tensor(miRNA)
            disease_th = th.Tensor(disease)
            # train_samples_th = th.Tensor(data['train_samples']).float()
            train_samples_th = th.Tensor(a).float()
            train_score, anchor_mm, positive_dd, negative_dd = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th,
                                                                     a, c)

            # Compute triplet loss
            triplet_loss = triplet_loss_fn(anchor_mm, positive_dd, negative_dd)
            # print(train_samples_th[:, 2])
            # print(th.flatten(train_score))
            # loss_mutual = loss_mutual_information(d1, d2, 64, 16) + loss_mutual_information(m1, m2, 64, 16)
            train_cross_loss = cross_entropy(th.flatten(train_score), train_samples_th[:, 2].to(device))

            train_loss = alpha * train_cross_loss + beta * triplet_loss
            # train_loss = train_cross_loss
            scoree, _, _, _ = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, b, c)
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


        scoree, _, _, _ = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, b, c)

        scoree = scoree.cpu()
        scoree = scoree.detach().numpy()

        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score
        import numpy as np

        # 获取 miRNA 和疾病的嵌入向量
        emb_mm = positive_dd.detach().cpu().numpy()  # miRNA embeddings
        emb_dd = negative_dd.detach().cpu().numpy()  # Disease embeddings

        # 合并嵌入向量（miRNA 和疾病）
        embeddings = np.concatenate((emb_mm, emb_dd), axis=0)

        # 获取标签：1表示AMP，0表示non-AMP
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
        # 存储每一折的结果
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)
        # 计算auc
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
        one_score = [aucc, auprc, accuracy, precision, recall, f1]
        all_score.append(one_score)
    cv_metric = np.mean(all_score, axis=0)
    sD_metric = np.std(all_score, axis=0)
    print('################5-Fold Result################')
    print_met(cv_metric)
    print_met2(sD_metric)
    # mean_fpr = np.linspace(0, 1, 100)
    # mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)], axis=0)
    # # mean_auc = auc(mean_fpr, mean_tpr)
    # plt.plot(mean_fpr, mean_tpr, 'k--', lw=2, label=f'Mean ROC curve (area = {cv_metric[0]:.4f})')
    #
    # # 绘制其他细节
    # plt.plot([0, 1], [0, 1], 'r--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC)')
    # plt.legend(loc='lower right')
    # # plt.grid(True)
    # output_directory = './result/Triplet_loss_ROC_cure_v3.2/'  # 替换为实际的文件夹路径
    # output_filename = f'{k}times_fivefold_ROC_curve.png'
    # output_path = f'{output_directory}/{output_filename}'
    # plt.savefig(output_path)
    # plt.close()
    # plt.show()
    # 创建主图
    # 创建主图
    # mean_fpr = np.linspace(0, 1, 100)
    # mean_tpr = np.mean([np.interp(mean_fpr, fpr_list[i], tpr_list[i]) for i in range(kfolds)], axis=0)
    #
    # # 创建主图
    # fig, ax = plt.subplots(figsize=(8, 6))
    #
    # # 绘制每一折的ROC曲线
    # colors = ['b', 'g', 'r', 'c', 'm']
    # for i in range(kfolds):
    #     ax.plot(fpr_list[i], tpr_list[i], lw=2, color=colors[i],
    #             label=f'Fold {i + 1} ROC (area = {roc_auc_list[i]:.4f})')
    #
    # # 绘制平均ROC曲线
    # ax.plot(mean_fpr, mean_tpr, color='k', linestyle='--', lw=1,
    #         label=f'Mean ROC (area = {cv_metric[0]:.4f})')
    #
    # # 绘制对角线
    # ax.plot([0, 1], [0, 1], 'r--', lw=1)
    #
    # # 设置主图的标签和标题
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.set_title('Receiver Operating Characteristic (ROC)')
    # ax.legend(loc='lower right')
    #
    # # 添加放大左上角区域的子图
    # axins = inset_axes(ax, width="35%", height="35%", loc='center right', borderpad=2)
    #
    # # 在子图中绘制所有ROC曲线
    # for i in range(kfolds):
    #     axins.plot(fpr_list[i], tpr_list[i], lw=1, color=colors[i])
    # # 绘制平均ROC曲线
    # axins.plot(mean_fpr, mean_tpr, color='k', linestyle='--', lw=2)
    #
    # # 设置子图的显示范围，放大左上角区域
    #
    # x1, x2 = 0.0, 0.3  # x 轴范围，从 0 到 0.2
    # y1, y2 = 0.8, 1.0  # y 轴范围，从 0.8 到 1.0
    # # x1, x2 = 0.0, 0.1  # 调整 x 轴范围，使虚线框更小
    # # y1, y2 = mean_tpr[mean_fpr <= x2].min(), mean_tpr[mean_fpr <= x2].max()
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    #
    # # 去掉子图的刻度标签
    # axins.tick_params(axis='both', which='both', length=0,labelsize=0)
    #
    # # 在主图上绘制表示放大区域的矩形框
    # rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
    #                      edgecolor='black', linestyle='--', facecolor='none', linewidth=1)
    # ax.add_patch(rect)
    #
    # # 使用 mark_inset 绘制虚线框和连接线
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", linestyle="--", linewidth=1)
    # output_directory = './result/Triplet_loss_ROC_cure_v3.2/'  # 替换为实际的文件夹路径
    # output_filename = f'{k}times_fivefold_ROC_curve.png'
    # output_path = f'{output_directory}/{output_filename}'
    # plt.savefig(output_path)
    # # 显示图形
    # # plt.show()
    # plt.close()
    #
    # mean_fpr = np.linspace(0, 1, 100)
    # mean_tpr = np.mean([np.interp(mean_fpr, fpr_list[i], tpr_list[i]) for i in range(kfolds)], axis=0)
    #
    # # 创建主图
    # fig, ax = plt.subplots(figsize=(8, 6))
    #
    # # 绘制每一折的ROC曲线
    # colors = ['b', 'g', 'r', 'c', 'm']
    # for i in range(kfolds):
    #     ax.plot(fpr_list[i], tpr_list[i], lw=2, color=colors[i],
    #             label=f'Fold {i + 1} ROC (area = {roc_auc_list[i]:.4f})')
    #
    # # 绘制平均ROC曲线
    # ax.plot(mean_fpr, mean_tpr, color='k', linestyle='--', lw=1,
    #         label=f'Mean ROC (area = {cv_metric[0]:.4f})')
    #
    # # 绘制对角线
    # ax.plot([0, 1], [0, 1], 'r--', lw=1)
    #
    # # 设置主图的标签和标题
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.set_title('Receiver Operating Characteristic (ROC)')
    # ax.legend(loc='lower right')
    #
    # # 添加放大左上角区域的子图
    # axins = inset_axes(ax, width="35%", height="35%", loc='center right', borderpad=2)
    #
    # # 在子图中绘制所有ROC曲线
    # for i in range(kfolds):
    #     axins.plot(fpr_list[i], tpr_list[i], lw=1, color=colors[i])
    # # 绘制平均ROC曲线
    # axins.plot(mean_fpr, mean_tpr, color='k', linestyle='--', lw=2)
    #
    # # 设置子图的显示范围，放大左上角区域
    # x1, x2 = 0.0, 0.3
    # y1, y2 = 0.8, 1.0
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    #
    # # 去掉子图的刻度标签
    # axins.tick_params(axis='both', which='both', length=0, labelsize=0)
    #
    # # 在主图上绘制表示放大区域的矩形框
    # rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
    #                      edgecolor='black', linestyle='--', facecolor='none', linewidth=1)
    # ax.add_patch(rect)
    #
    # # 使用 mark_inset 绘制虚线框和连接线
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", linestyle="--", linewidth=1)
    #
    # # 添加连接线
    # ax.plot([x1, x1], [y1, y1 + 0.1], color='black', linestyle='--', lw=1)
    #
    # # 子图边框设置为虚线
    # for spine in axins.spines.values():
    #     spine.set_edgecolor('black')
    #     spine.set_linestyle('--')
    #
    # output_directory = './result/ncRNA_Drug/'  # 替换为实际的文件夹路径
    # output_filename = f'{k}times_fivefold_ROC_curve.png'
    # output_path = f'{output_directory}/{output_filename}'
    # plt.savefig(output_path)
    # # plt.show()
    # plt.close()
    # # 假设已定义相关数据：fpr_list, tpr_list, roc_auc_list, cv_metric, kfolds



    #Final plot
    # mean_fpr = np.linspace(0, 1, 100)
    # mean_tpr = np.mean([np.interp(mean_fpr, fpr_list[i], tpr_list[i]) for i in range(kfolds)], axis=0)
    #
    # # 创建主图
    # fig, ax = plt.subplots(figsize=(8, 6))
    #
    # # 绘制每一折的ROC曲线
    # colors = ['b', 'g', 'r', 'c', 'm']
    # for i in range(kfolds):
    #     ax.plot(fpr_list[i], tpr_list[i], lw=2, color=colors[i],
    #             label=f'Fold {i + 1} ROC (area = {roc_auc_list[i]:.4f})')
    #
    # # 绘制平均ROC曲线
    # ax.plot(mean_fpr, mean_tpr, color='k', linestyle='--', lw=1,
    #         label=f'Mean ROC (area = {cv_metric[0]:.4f})')
    #
    # # 绘制对角线
    # ax.plot([0, 1], [0, 1], 'r--', lw=1)
    #
    # # 设置主图的标签和标题
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.set_title('Receiver Operating Characteristic (ROC)')
    # ax.legend(loc='lower right')
    #
    # # 添加放大左上角区域的子图
    # axins = inset_axes(ax, width="35%", height="30%", loc='center right', borderpad=3)
    #
    # # 在子图中绘制所有ROC曲线
    # for i in range(kfolds):
    #     axins.plot(fpr_list[i], tpr_list[i], lw=1, color=colors[i])
    # # 绘制平均ROC曲线
    # axins.plot(mean_fpr, mean_tpr, color='k', linestyle='--', lw=1)
    #
    # # 设置子图的显示范围，放大左上角区域
    # x1, x2 = 0.0, 0.3
    # y1, y2 = 0.8, 1.0
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    #
    # # 去掉子图的刻度标签
    # axins.tick_params(axis='both', which='both')
    #
    # # 1 (右上) 2 (左上) 3(左下) 4(右下)
    # # 在主图上绘制表示放大区域的矩形框
    # rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='black', linestyle='--', facecolor='none', linewidth=1)
    # ax.add_patch(rect)
    #
    # # 使用 mark_inset 绘制虚线框和连接线
    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="black", linestyle="--", linewidth=1)
    #
    # # 修改子图边框为虚线
    # for spine in axins.spines.values():
    #     spine.set_linestyle('--')
    #     spine.set_linewidth(1)
    #
    # # 保存图像
    # output_directory = './result/ncRNA_Drug/'  # 替换为实际的文件夹路径 #Triplet_loss_PR_curve_v2.0
    # # output_filename = f'{k}times_fivefold_ROC_curve.png'
    # output_filename = f'fivefold_ROC_curve.png'
    # output_path = f'{output_directory}/{output_filename}'
    # plt.savefig(output_path)
    # # plt.show()
    # plt.close()
    #
    # mean_recall = np.linspace(0, 1, 100)
    # mean_precision = np.zeros_like(mean_recall)
    # for i in range(kfolds):
    #     # 因为 recall 是降序的，所以需要翻转
    #     precision_interp = np.interp(mean_recall, recall_list[i][::-1], precision_list[i][::-1])
    #     mean_precision += precision_interp
    # mean_precision /= kfolds
    #
    # # 创建PR曲线图
    # fig, ax = plt.subplots(figsize=(8, 6))
    #
    # # 绘制每一折的PR曲线
    # colors = ['b', 'g', 'r', 'c', 'm']
    # for i in range(kfolds):
    #     ax.plot(recall_list[i], precision_list[i], lw=1, color=colors[i],
    #             label=f'Fold {i + 1} PR (area = { auprc_list[i]:.4f})')
    #
    # # 绘制平均PR曲线
    # ax.plot(mean_recall, mean_precision, color='k', linestyle='--', lw=2,
    #         label=f'Mean PR (area = {cv_metric[1]:.4f})')
    #
    # # 设置坐标轴和标题
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # ax.set_xlabel('Recall')
    # ax.set_ylabel('Precision')
    # ax.set_title('Precision-Recall Curve')
    # ax.legend(loc='lower left')
    #
    # # 添加放大右上角区域的子图
    # axins = inset_axes(ax, width="35%", height="30%", loc='center left', borderpad=3)
    #
    # # 在子图中绘制所有PR曲线
    # for i in range(kfolds):
    #     axins.plot(recall_list[i], precision_list[i], lw=1, color=colors[i])
    # # 绘制平均PR曲线
    # axins.plot(mean_recall, mean_precision, color='k', linestyle='--', lw=1)
    #
    # # 设置子图的显示范围，放大右上角区域
    # x1, x2 = 0.7, 1.0  # Recall范围
    # y1, y2 = 0.7, 1.0  # Precision范围
    #
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    #
    # # 去掉子图的刻度标签
    # axins.tick_params(axis='both', which='both')
    #
    # # 在主图上绘制表示放大区域的矩形框
    # rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
    #                      edgecolor='black', linestyle='--', facecolor='none', linewidth=1)
    # ax.add_patch(rect)
    #
    # # 使用 mark_inset 绘制虚线框和连接线
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", linestyle="--", linewidth=1)
    # output_directory = './result/ncRNA_Drug/'  # Replace with actual directory
    # # output_filename = f'{k}times_fivefold_PR_curve.png'
    # output_filename = f'fivefold_PR_curve.png'
    # output_path = f'{output_directory}/{output_filename}'
    #
    # plt.savefig(output_path)
    # # 显示图形
    # # plt.show()
    # plt.close()
    #
    # return cv_metric, sD_metric

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    # import numpy as np
    #
    # # 假设这些数据已经通过模型训练和交叉验证计算出来了：
    # # fpr_list, tpr_list, roc_auc_list, recall_list, precision_list, auprc_list, cv_metric, kfolds
    #
    # # 创建包含一行两列的子图
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    #
    # # ROC 曲线
    # mean_fpr = np.linspace(0, 1, 100)
    # mean_tpr = np.mean([np.interp(mean_fpr, fpr_list[i], tpr_list[i]) for i in range(kfolds)], axis=0)
    #
    # # 绘制每一折的ROC曲线
    # colors = ['b', 'g', 'r', 'c', 'm']
    # for i in range(kfolds):
    #     ax1.plot(fpr_list[i], tpr_list[i], lw=2, color=colors[i],
    #              label=f'Fold {i + 1} ROC (area = {roc_auc_list[i]:.4f})')
    #
    # # 绘制平均ROC曲线
    # ax1.plot(mean_fpr, mean_tpr, color='k', linestyle='--', lw=1, label=f'Mean ROC (area = {cv_metric[0]:.4f})')
    #
    # # 绘制对角线
    # # ax1.plot([0, 1], [0, 1], 'r--', lw=1)
    #
    # # 设置ROC曲线图的标签和标题
    # ax1.set_xlim([0.0, 1.0])
    # ax1.set_ylim([0.0, 1.05])
    # ax1.set_xlabel('False Positive Rate')
    # ax1.set_ylabel('True Positive Rate')
    # # ax1.set_title('(C) ROC curve of MIFNDRA on dataset2 during five-fold cross-validation')
    # ax1.legend(loc='lower right')
    #
    # # 放大左上角区域的子图
    # axins = inset_axes(ax1, width="35%", height="30%", loc='center right', borderpad=3)
    # for i in range(kfolds):
    #     axins.plot(fpr_list[i], tpr_list[i], lw=1, color=colors[i])
    # axins.plot(mean_fpr, mean_tpr, color='k', linestyle='--', lw=1)
    #
    # x1, x2 = 0.0, 0.3
    # y1, y2 = 0.8, 1.0
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.tick_params(axis='both', which='both')
    # rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='black', linestyle='--', facecolor='none', linewidth=1)
    # ax1.add_patch(rect)
    # mark_inset(ax1, axins, loc1=3, loc2=1, fc="none", ec="black", linestyle="--", linewidth=1)
    # for spine in axins.spines.values():
    #     spine.set_linestyle('--')
    #     spine.set_linewidth(1)
    #
    # # PR 曲线
    # mean_recall = np.linspace(0, 1, 100)
    # mean_precision = np.zeros_like(mean_recall)
    # for i in range(kfolds):
    #     precision_interp = np.interp(mean_recall, recall_list[i][::-1], precision_list[i][::-1])
    #     mean_precision += precision_interp
    # mean_precision /= kfolds
    #
    # # 绘制每一折的PR曲线
    # for i in range(kfolds):
    #     ax2.plot(recall_list[i], precision_list[i], lw=1, color=colors[i],
    #              label=f'Fold {i + 1} PR (area = {auprc_list[i]:.4f})')
    #
    # # 绘制平均PR曲线
    # ax2.plot(mean_recall, mean_precision, color='k', linestyle='--', lw=2, label=f'Mean PR (area = {cv_metric[1]:.4f})')
    #
    # # 设置PR曲线图的标签和标题
    # ax2.set_xlim([0.0, 1.0])
    # ax2.set_ylim([0.0, 1.05])
    # ax2.set_xlabel('Recall')
    # ax2.set_ylabel('Precision')
    # ax2.set_title('Precision-Recall Curve')
    # ax2.legend(loc='lower left')
    #
    # # 放大右上角区域的子图
    # axins2 = inset_axes(ax2, width="35%", height="30%", loc='center left', borderpad=3)
    # for i in range(kfolds):
    #     axins2.plot(recall_list[i], precision_list[i], lw=1, color=colors[i])
    # axins2.plot(mean_recall, mean_precision, color='k', linestyle='--', lw=1)
    #
    # x1, x2 = 0.7, 1.0
    # y1, y2 = 0.7, 1.0
    # axins2.set_xlim(x1, x2)
    # axins2.set_ylim(y1, y2)
    # axins2.tick_params(axis='both', which='both')
    # rect2 = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='black', linestyle='--', facecolor='none', linewidth=1)
    # ax2.add_patch(rect2)
    # mark_inset(ax2, axins2, loc1=2, loc2=4, fc="none", ec="black", linestyle="--", linewidth=1)
    #
    # # 保存图像
    # output_directory = './result/ncRNA_Drug/'  # 替换为实际的文件夹路径
    # output_filename = 'fivefold_ROC_PR_curve1.png'
    # output_path = f'{output_directory}/{output_filename}'
    # plt.savefig(output_path)
    #
    # # 关闭绘图
    # plt.close()

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

