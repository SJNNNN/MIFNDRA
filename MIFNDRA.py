import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.nn.pytorch
import torch as th
from torch import nn,einsum
from dgl import function as fn
from dgl.nn import pytorch as pt

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
class NonlinearFusion(nn.Module):
    def __init__(self, input_dim):
        super(NonlinearFusion, self).__init__()
        self.fc1 = nn.Linear(2 * input_dim, input_dim,device=device)
        self.fc2 = nn.Linear(input_dim, 1,device=device)

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=-1)
        attention = torch.sigmoid(self.fc2(F.relu(self.fc1(combined))))
        return attention * x1 + (1 - attention) * x2

class GlobalFusion(nn.Module):
    def __init__(self, input_dim):
        super(GlobalFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim,device=device)
        self.fc2 = nn.Linear(input_dim, 1,device=device)

    def forward(self, x):
        global_context = torch.mean(x, dim=0, keepdim=True)
        attention = torch.sigmoid(self.fc2(F.relu(self.fc1(global_context))))
        return attention * x + (1 - attention) * global_context

class MIFNDRA(nn.Module):
    def __init__(self, args):
        super(MIFNDRA, self).__init__()
        self.args = args
        self.lin_m = nn.Linear(args.miRNA_number, args.in_feats, bias=False)
        self.lin_d = nn.Linear(args.disease_number, args.in_feats, bias=False)


        self.gcn_mm_1 = dglnn.SAGEConv(args.miRNA_number, 128, 'mean')
        self.gcn_mm_2 = dglnn.SAGEConv(128, 64, 'mean')
        self.gcn_mm_3 = dglnn.SAGEConv(64, args.out_feats, 'mean')
        self.res_l_1 = nn.Linear(args.miRNA_number, 64)

        self.gcn_dd_1 = dglnn.SAGEConv(args.disease_number, 128, 'mean')
        self.gcn_dd_2 = dglnn.SAGEConv(128, 64, 'mean')
        self.gcn_dd_3 = dglnn.SAGEConv(64, args.out_feats, 'mean')
        self.res_l_2 = nn.Linear(args.disease_number, 64)

        self.gcn_md_1 = dglnn.SAGEConv(args.in_feats,128, 'mean')
        self.gcn_md_2 = dglnn.SAGEConv(128, 64, 'mean')
        self.gcn_md_3 = dglnn.SAGEConv(64, args.out_feats, 'mean')
        self.res_l_3 = nn.Linear(args.in_feats, args.out_feats)

        self.elu = nn.ELU()
        self.mlp = nn.Sequential()
        self.dropout = nn.Dropout(args.dropout)
        in_feat =2* args.out_feats

        for idx, out_feat in enumerate(args.mlp):
            if idx == 0:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('elu', nn.ELU())
                self.mlp.add_module('dropout', nn.Dropout(p=0.2))
                in_feat = out_feat
            else:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('sigmoid', nn.Sigmoid())
                self.mlp.add_module('dropout', nn.Dropout(p=0.2))
                in_feat = out_feat

        self.NonlinearFusion_m = NonlinearFusion(args.out_feats)
        self.NonlinearFusion_d = NonlinearFusion(args.out_feats)
        self.NonlinearFusion_md = NonlinearFusion(args.out_feats)
        self.global_fusion_m = GlobalFusion(args.out_feats)
        self.global_fusion_d = GlobalFusion(args.out_feats)
        self.global_fusion_md = GlobalFusion(args.out_feats)

    def forward(self, mm_graph, dd_graph, md_graph, miRNA, disease, samples,triplet_samples):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mm_graph = mm_graph.to(device)
        dd_graph = dd_graph.to(device)
        md_graph = md_graph.to(device)
        miRNA = miRNA.to(device)
        disease = disease.to(device)

        res_mi = self.elu(self.res_l_1(miRNA))
        res_di = self.elu(self.res_l_2(disease))

        md = torch.cat((self.lin_m(miRNA), self.lin_d(disease)), dim=0)
        res_md = self.elu(self.res_l_3(md))

        res_mm = res_md[:self.args.miRNA_number, :]
        res_dd = res_md[self.args.miRNA_number:, :]
        res_mmdd = torch.cat((res_mm, res_dd), dim=0)

        emb_mm_sim_1 = self.elu(self.gcn_mm_1(mm_graph, miRNA))
        emb_mm_sim_2 = self.elu(self.gcn_mm_2(mm_graph, emb_mm_sim_1))
        emb_mm_sim_3 = self.elu(self.gcn_mm_3(mm_graph, emb_mm_sim_2))
        emb_mm_sim_3 = self.NonlinearFusion_m(emb_mm_sim_3, res_mi)

        emb_dd_sim_1 = self.elu(self.gcn_dd_1(dd_graph, disease))
        emb_dd_sim_2 = self.elu(self.gcn_dd_2(dd_graph, emb_dd_sim_1))
        emb_dd_sim_3 = self.elu(self.gcn_dd_3(dd_graph, emb_dd_sim_2))
        emb_dd_sim_3 = self.NonlinearFusion_d(emb_dd_sim_3, res_di)

        emb_ass_1 = self.elu(self.gcn_md_1(md_graph, torch.cat((self.lin_m(miRNA), self.lin_d(disease)), dim=0)))
        emb_ass_2 = self.elu(self.gcn_md_2(md_graph, emb_ass_1))
        emb_ass_3 = self.elu(self.gcn_md_3(md_graph, emb_ass_2))
        emb_ass_3 = self.NonlinearFusion_md(emb_ass_3, res_mmdd)

        emb_mm_ass = emb_ass_3[:self.args.miRNA_number, :]
        emb_dd_ass = emb_ass_3[self.args.miRNA_number:, :]

        emb_mm = self.global_fusion_m(self.NonlinearFusion_m(emb_mm_sim_3, emb_mm_ass))
        emb_dd = self.global_fusion_d(self.NonlinearFusion_d(emb_dd_sim_3, emb_dd_ass))


        anchor_mm = emb_mm[triplet_samples[:, 0]]
        positive_dd = emb_dd[triplet_samples[:, 1]]
        negative_dd = emb_dd[triplet_samples[:, 2]]
        # print(emb_mm[samples[:, 0]].shape)
        # print( emb_dd[samples[:, 1]].shape)
        # Classification outputs
        emb = torch.cat((emb_mm[samples[:, 0]], emb_dd[samples[:, 1]]), dim=1)
        classification_output = self.mlp( emb )

        return classification_output, anchor_mm, positive_dd, negative_dd




