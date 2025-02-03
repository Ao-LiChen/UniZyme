import torch
import esm
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn.functional as F
from scipy.spatial import distance_matrix
from Bio.PDB import PDBParser
import numpy as np
import os
import torch.nn as nn
import math
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import random

from pytorchtools import EarlyStopping
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0,2,3,4,5,6,7"
# Parameters
distance_threshold = 8.0
num_epochs = 10
learning_rate = 0.0001
real_batch_size = 40  # 每次批量加载 ESM 特征
train_ratio = 0.8  # 80% of data for training
num_attention_heads = 2
num_transformer_layers = 1  # 多层 Attention
K=10

random.seed(42)


class SequenceDataset(Dataset):
    def __init__(self, data, uniprot_dict, mernum_dict):
        self.data = data
        self.uniprot_dict = uniprot_dict
        self.mernum_dict = mernum_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据
        entry = self.data[idx]
        key = entry[0]

        # 提取 Uniprot 和 Mernum 序列
        uniprot_name, mernum_name = key.split('_')
        batch_size = 1

        uniprot_seq = self.uniprot_dict.get(uniprot_name, '')
        mernum_seq = self.mernum_dict.get(mernum_name, '')[1]

        data = [(uniprot_name, uniprot_seq)]
        data.extend([(mernum_name, mernum_seq)])
        batch_labels, batch_strs, batch_tokens = batch_converter(data)


        batch_lens = [len(seq) for seq in batch_strs]

        padded_distances = []
        padded_energy = []
        activate_padded_labels = []

        max_len = 1500

        success=[]

        for i, seq_len in enumerate(batch_lens):
            if batch_labels[i].startswith("M"):
                pdb_file = f"../data/Enzyme_Structure/{batch_labels[i]}.pdb"
                try:
                    energy = pickle.load(open(f"../data/Enzyme_Energy/{batch_labels[i]}_configurational.pkl","rb"))
                except:
                    return None

                energy = torch.tensor(energy, dtype=torch.float)

            else:
                energy = torch.zeros(max_len, max_len, dtype=torch.float)
                pdb_file = f"../data/Substrate_Structures/{batch_labels[i]}.pdb"

            try:
                distances = create_distance(pdb_file)
            except:
                return None

            success.append(i)

            # esm_features_padded = esm_features_padded.view(1, esm_features_padded.size(0), esm_features_padded.size(1))

            # 填充距离矩阵和能量矩阵
            if seq_len < max_len:
                pad_distance = torch.zeros(seq_len, max_len - seq_len)
                pad_distance_row = torch.zeros(max_len - seq_len, max_len)
                distances_padded = torch.cat([distances, pad_distance], dim=1)
                distances_padded = torch.cat([distances_padded, pad_distance_row], dim=0)  # (max_len, max_len)

                if batch_labels[i].startswith("M"):
                    energy_padded = torch.cat([energy, pad_distance], dim=1)
                    energy_padded = torch.cat([energy_padded, pad_distance_row], dim=0)  # (max_len, max_len)
                else:
                    energy_padded = energy
            else:
                distances_padded = distances
                energy_padded = energy

            distances_padded = distances_padded.unsqueeze(0)  # 增加 batchsize 和 head 维度
            distances_padded = distances_padded.expand(num_attention_heads, -1, -1)

            energy_padded = energy_padded.unsqueeze(0)  # 增加 batchsize 和 head 维度
            energy_padded = energy_padded.expand(num_attention_heads, -1, -1)

            energy_padded = energy_padded
            distances_padded = distances_padded

            if batch_labels[i].startswith("M"):
                site_indices = enzyme_dict[batch_labels[i]][0]
                labels = torch.zeros(max_len, dtype=torch.float)
                site_indices = [x - 1 for x in site_indices]
                labels[site_indices] = 1.0
                activate_padded_labels.append(labels)


            padded_distances.append(distances_padded)
            padded_energy.append(energy_padded)

        success_real = []
        for flag in range(batch_size):
            if (flag in success and flag+batch_size in success):
                success_real.append(flag)

        if success_real==[]:
            return None

        # 使用success_real对数据进行二次筛选
        padded_distances_real = []
        padded_energy_real = []
        activate_padded_labels_real = []
        padded_cleavage_labels_real = []

        for each in success_real:

            padded_distances_real.append(padded_distances[each])

            padded_energy_real.append(padded_energy[each])

            activate_padded_labels_real.append(activate_padded_labels[each])


            name = uniprot_name + "_" + mernum_name

            try:
                cleavage_site = train_dataset[name]
            except:
                try:
                    cleavage_site = test_dataset[name]
                except:
                    cleavage_site = validation_dataset[name]


            cleavage_site=[x-1 for x in cleavage_site]
            cleavage_labels = torch.zeros(max_len, dtype=torch.float)
            cleavage_labels[cleavage_site] = 1.0
            padded_cleavage_labels_real.append(cleavage_labels)

        for each in success_real:
            padded_distances_real.append(padded_distances[each + batch_size])
            padded_energy_real.append(padded_energy[each + batch_size])


        distances_batch = torch.stack(padded_distances_real)
        energy_batch = torch.stack(padded_energy_real)
        activate_labels_batch = torch.stack(activate_padded_labels_real)
        cleavage_labels_batch = torch.stack(padded_cleavage_labels_real)


        mask = torch.zeros(len(distances_batch), num_attention_heads, max_len, max_len)  # 初始化掩码为全0

        for i,index in enumerate(success_real):
            sub_seq_len = batch_lens[index]
            enzyme_seq_len= batch_lens[index+batch_size]
            mask[i, :, :sub_seq_len, :sub_seq_len] = 1.0
            mask[i+batch_size, :, :enzyme_seq_len, :enzyme_seq_len] = 1.0


        Enzymes_distances = distances_batch[1]
        Substrate_distances = distances_batch[0]
        Enzymes_energy = energy_batch[1]
        Enzymes_mask = mask[1]
        Substrate_mask = mask[0]

        padto=1502
        batch_tokens_uniprot= batch_tokens[0]
        batch_tokens_uniprot = F.pad(batch_tokens_uniprot, (0, padto - batch_tokens_uniprot.size(0)), value=1)
        batch_tokens_enzyme = batch_tokens[1]
        batch_tokens_enzyme = F.pad(batch_tokens_enzyme, (0, padto - batch_tokens_enzyme.size(0)), value=1)

        batch_lens_uniprot = batch_lens[0]
        batch_lens_enzyme = batch_lens[1]



        return (batch_tokens_uniprot,batch_tokens_enzyme,batch_lens_uniprot,batch_lens_enzyme,Enzymes_distances,Substrate_distances,Enzymes_energy,Enzymes_mask,Substrate_mask,activate_labels_batch,cleavage_labels_batch)


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Function to parse PDB file and extract CA atom coordinates
def get_ca_coordinates(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    ca_coords = [residue['CA'].coord for model in structure for chain in model for residue in chain if 'CA' in residue]
    return np.array(ca_coords)


# Create node features and distance matrix
def create_distance(pdb_file):
    ca_coords = get_ca_coordinates(pdb_file)
    num_nodes = ca_coords.shape[0]
    distances = distance_matrix(ca_coords, ca_coords)
    # 倒数
    distances = 1 / (distances + 1)
    distances = torch.tensor(distances, dtype=torch.float)
    # distances[distances > distance_threshold] = 0  # Apply threshold
    return distances


class GaussianParameters(nn.Module):
    def __init__(self, K=10):
        super(GaussianParameters, self).__init__()
        self.K = K
        self.mu_D_Enzyme = nn.Parameter(torch.randn(K))        # 距离高斯核中心
        self.sigma_D_Enzyme = nn.Parameter(torch.ones(K))      # 距离高斯核标准差
        self.b_D_Enzyme = nn.Parameter(torch.zeros(K))         # 距离偏置项

        self.mu_D_Sub = nn.Parameter(torch.randn(K))        # 距离高斯核中心
        self.sigma_D_Sub = nn.Parameter(torch.ones(K))      # 距离高斯核标准差
        self.b_D_Sub = nn.Parameter(torch.zeros(K))         # 距离偏置项

        self.mu_E = nn.Parameter(torch.randn(K))        # 能量高斯核中心
        self.sigma_E = nn.Parameter(torch.ones(K))      # 能量高斯核标准差
        self.b_E = nn.Parameter(torch.zeros(K))         # 能量偏置项

        # 定义用于Phi计算的线性层（公式5）
        self.W_D1 = nn.Linear(K, K)  # [K x K]
        self.W_D2 = nn.Linear(K, 1)  # [K x 1]

        self.mu_activation = nn.Parameter(torch.randn(K))        # 激活位点高斯核中心
        self.sigma_activation = nn.Parameter(torch.ones(K))      # 激活位点高斯核标准差
        self.b_activation = nn.Parameter(torch.zeros(K))         # 激活位点偏置项
        # 定义用于Phi计算的线性层（公式5）
        self.W_activation1 = nn.Linear(K, K)  # [K x K]
        self.W_activation2 = nn.Linear(K, 1)  # [K x 1]



class Pool_Cat_MLP(torch.nn.Module):
    def __init__(self, d, gaussian_params):
        super(Pool_Cat_MLP,self).__init__()
        self.d = d
        self.mlp = nn.Sequential(
            nn.Linear(2*d, d),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(d, 64),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        #一维卷积
        #self.conv1d = nn.Conv1d(in_channels=2 * d, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1d = nn.Conv1d(in_channels=2*d, out_channels=128, kernel_size=31, stride=1, padding=15)
        # 定义激活函数和 Dropout
        self.conv_activation = nn.SELU()
        self.conv_dropout = nn.Dropout(p=0.2)

        # 定义全连接层，用于最终的预测
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.gaussian_params = gaussian_params

    def forward(self, Enzyme, Substrate, Activate_Site,Activate_Site_f,Enzymes_mask):
        #softmax
        # softmax_lambda_weight = F.softmax(lambda_weight, dim=-1)
        # #对Enzyme基于权重的pool
        # weighted_Enzyme = softmax_lambda_weight * Enzyme
        #pooled_Enzyme = weighted_Enzyme.sum(dim=-2)
        #
        # #平均池化
        # Enzymes_mask = Enzymes_mask[:, 0, :, 0]
        # #根据mask加权Enzyme
        # Enzymes_length = Enzymes_mask.sum(dim=-1)
        # Enzymes_mask = Enzymes_mask.unsqueeze(-1).repeat(1, 1, self.d)
        # pooled_Enzyme = (Enzyme * Enzymes_mask).sum(dim=-2) / Enzymes_length.view(-1, 1)


        # 平均池化
        Enzymes_mask = Enzymes_mask[:, 0, :, 0]
        # 根据mask加权Enzyme
        Enzymes_length = Enzymes_mask.sum(dim=-1)
        Enzymes_mask = Enzymes_mask.unsqueeze(-1).repeat(1, 1, self.d)

        Activate_Site_f = Activate_Site_f.repeat(1, 1, self.gaussian_params.K)
        Activate_Site_f = Activate_Site_f + self.gaussian_params.b_activation

        psi_Activate_Site_f = torch.exp(
            -0.5 * ((Activate_Site_f - self.gaussian_params.mu_activation) / self.gaussian_params.sigma_activation) ** 2)
        psi_Activate_Site_f = psi_Activate_Site_f / (
                torch.sqrt(torch.tensor(2 * math.pi, device=psi_Activate_Site_f.device)) * self.gaussian_params.sigma_activation)

        # 计算Phi_energy_enzyme（公式5）
        # psi_energy_enzyme_flat = psi_energy_enzyme.view(-1, self.K)  # [B*H*S*S, K]
        psi_Activate_Site_f_flat = psi_Activate_Site_f
        psi_Activate_Site_f_flat = F.gelu(self.gaussian_params.W_activation1(psi_Activate_Site_f_flat))  # [B*H*S*S, K]
        psi_Activate_Site_f = self.gaussian_params.W_activation2(psi_Activate_Site_f_flat)  # [B*H*S*S, 1]
        # Phi_energy_enzyme = Phi_energy_enzyme_flat.view(l//2, n_heads, seq_len, seq_len)  # [batch_enzyme, n_heads, seq, seq]


        flag = psi_Activate_Site_f * Enzymes_mask


        # softmax
        flag = flag.masked_fill(flag == 0.0, -1e9)
        softmax_lambda_weight = F.softmax(flag, dim=-2)

        pooled_Enzyme = (softmax_lambda_weight * Enzyme).sum(dim=-2)


        pooled_Enzyme = pooled_Enzyme.unsqueeze(1).repeat(1, Substrate.size(1), 1)


        combined_features = torch.cat((Substrate,pooled_Enzyme), dim=-1)
        # prediction = self.mlp(combined_features)

        #下面进行一维卷积，Enzyme在Substrate上滑动
        #combined_features = Substrate + pooled_Enzyme  # 形状: (batch, length_sub, d)

        # 转换为 Conv1d 需要的形状: (batch, d, length_sub)
        combined_features = combined_features.permute(0, 2, 1)

        # 应用一维卷积
        conv_output = self.conv1d(combined_features)  # (batch, 64, length_sub)
        conv_output = self.conv_activation(conv_output)
        conv_output = self.conv_dropout(conv_output)

        # 转换回 (batch, length_sub, 64)
        conv_output = conv_output.permute(0, 2, 1)

        # 通过全连接层进行预测
        prediction = self.fc(conv_output)  # (batch, length_sub, 1)

        return prediction

#
# class CrossAttentionWithWeight(torch.nn.Module):
#     def __init__(self, d):
#         super(CrossAttentionWithWeight, self).__init__()
#         self.d = d
#         # 构建MLP
#         self.linerq = nn.Sequential(nn.Linear(d, d))
#         self.linerk = nn.Sequential(nn.Linear(d, d))
#         self.linerv = nn.Sequential(nn.Linear(d, d))
#         self.mlp = nn.Sequential(
#             nn.Linear(d, 128),
#             nn.Dropout(p=0.3),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.Dropout(p=0.3),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Dropout(p=0.3),
#             nn.Sigmoid()
#         )
#
#     def forward(self, Q, K, V, lambda_weight):
#
#         Q = self.linerq(Q)
#         K = self.linerk(K)
#         V = self.linerv(V)
#
#
#
#         # 计算查询和键的点积
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # 点积，大小 [N, M]
#         # 缩放并应用可学习的调节因子
#         attention_scores = attention_scores / (self.d ** 0.5)  # 缩放
#
#
#         attention_scores *= lambda_weight  # 应用调节因子
#         # 计算注意力权重
#         attention_weights = F.softmax(attention_scores, dim=-1)  # softmax，大小 [N, M]
#         # 计算最终输出
#         output = torch.matmul(attention_weights, V)  # 加权求和，输出大小 [N, d_v]
#
#         prediction = self.mlp(output)
#
#         return prediction, attention_weights
#
#



# Define the Transformer model with multi-layer attention
class ScaledDotProductAttention_Gauss(nn.Module):
    def __init__(self, d_model,gaussian_params, K=10):
        super(ScaledDotProductAttention_Gauss, self).__init__()
        self.d_model = d_model
        self.K = K  # 高斯核的数量

        self.gaussian_params = gaussian_params


        # self.adapter = nn.Sequential(
        #     nn.Linear(1500, 1500),
        #     nn.ReLU(),
        #     nn.Linear(1500, 1500)
        # )  # Define a Linear-based adapter module


    def forward(self, Q, K, V, distance_matrix, energy_matrix, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_model * 2)  # scores : [batch_size, n_heads, seq_len, seq_len]
        l= scores.size(0)
        batch_size, n_heads, seq_len, _ = scores.shape

        distance_matrix = distance_matrix.unsqueeze(-1).repeat(1,1, 1, 1, self.K)
        #print(distance_matrix.shape)

        if energy_matrix is None:
            # 计算距离带偏置
            distance = distance_matrix + self.gaussian_params.b_D_Sub  # [batch_sub, n_heads, seq, seq]
            # 计算psi_dist_substrate（公式1）
            # [batch_sub, n_heads, seq, seq, K]
            psi_dist = torch.exp(-0.5 * ((distance - self.gaussian_params.mu_D_Sub) / self.gaussian_params.sigma_D_Sub) ** 2)
            psi_dist = psi_dist / (
                        torch.sqrt(torch.tensor(2 * math.pi, device=distance.device)) * self.gaussian_params.sigma_D_Sub)
            # 计算Phi_dist_substrate（公式5）
            # 将psi_dist_substrate展平以通过线性层
            #psi_dist_substrate_flat = psi_dist_substrate.view(-1, self.K)  # [B*H*S*S, K]
            psi_dist_flat = psi_dist
            psi_dist_flat = F.gelu(self.gaussian_params.W_D1(psi_dist_flat))  # [B*H*S*S, K]
            psi_dist_flat = self.gaussian_params.W_D2(psi_dist_flat)  # [B*H*S*S, 1]
            #Phi_dist_substrate = Phi_dist_substrate_flat.view(l//2, n_heads, seq_len, seq_len)  # [batch_sub, n_heads, seq, seq]

            Phi_dist = psi_dist_flat.squeeze(-1)  # [batch_sub, n_heads, seq, seq]



            # 应用ReLU和适配器层（公式7）
            #Phi_dist = self.adapter(Phi_dist)

            # 将Phi_dist_substrate添加到scores_substrate（公式6）
            scores = scores + Phi_dist  # [batch_sub, n_heads, seq, seq]
        else:
            # 计算距离带偏置
            distance = distance_matrix + self.gaussian_params.b_D_Enzyme  # [batch_sub, n_heads, seq, seq]
            # 计算psi_dist_substrate（公式1）
            # [batch_sub, n_heads, seq, seq, K]
            psi_dist = torch.exp(
                -0.5 * ((distance - self.gaussian_params.mu_D_Enzyme) / self.gaussian_params.sigma_D_Enzyme) ** 2)
            psi_dist = psi_dist / (
                    torch.sqrt(torch.tensor(2 * math.pi, device=distance.device)) * self.gaussian_params.sigma_D_Enzyme)
            # 计算Phi_dist_substrate（公式5）
            # 将psi_dist_substrate展平以通过线性层
            # psi_dist_substrate_flat = psi_dist_substrate.view(-1, self.K)  # [B*H*S*S, K]
            psi_dist_flat = psi_dist
            psi_dist_flat = F.gelu(self.gaussian_params.W_D1(psi_dist_flat))  # [B*H*S*S, K]
            psi_dist_flat = self.gaussian_params.W_D2(psi_dist_flat)  # [B*H*S*S, 1]
            # Phi_dist_substrate = Phi_dist_substrate_flat.view(l//2, n_heads, seq_len, seq_len)  # [batch_sub, n_heads, seq, seq]

            Phi_dist = psi_dist_flat.squeeze(-1)  # [batch_sub, n_heads, seq, seq]



            energy_matrix = energy_matrix.unsqueeze(-1).repeat(1, 1, 1, 1, self.K)
            energy = energy_matrix + self.gaussian_params.b_E

            psi_energy = torch.exp(
                -0.5 * ((energy - self.gaussian_params.mu_E) / self.gaussian_params.sigma_E) ** 2)
            psi_energy = psi_energy / (
                    torch.sqrt(torch.tensor(2 * math.pi, device=psi_energy.device)) * self.gaussian_params.sigma_E)

            # 计算Phi_energy_enzyme（公式5）
            # psi_energy_enzyme_flat = psi_energy_enzyme.view(-1, self.K)  # [B*H*S*S, K]
            psi_energy_flat = psi_energy
            psi_energy_flat = F.gelu(self.gaussian_params.W_D1(psi_energy_flat))  # [B*H*S*S, K]
            psi_energy_flat = self.gaussian_params.W_D2(psi_energy_flat)  # [B*H*S*S, 1]
            # Phi_energy_enzyme = Phi_energy_enzyme_flat.view(l//2, n_heads, seq_len, seq_len)  # [batch_enzyme, n_heads, seq, seq]
            Phi_energy = psi_energy_flat.squeeze(-1)  # [batch_enzyme, n_heads, seq, seq]

            # 将Phi_dist_enzyme和Phi_energy_enzyme添加到scores_enzyme（公式6）
            scores = scores + Phi_dist + Phi_energy  # [batch_enzyme, n_heads, seq, seq


        scores = scores.masked_fill(mask == 0, -1e9)

        attn = nn.Softmax(dim=-1)(scores)

        context = torch.matmul(attn, V)
        return context




# Define the Transformer model with multi-layer attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model

        # self.adapter = nn.Sequential(
        #     nn.Linear(1500, 1500),
        #     nn.ReLU(),
        #     nn.Linear(1500, 1500)
        # )  # Define a Linear-based adapter module

    def forward(self, Q, K, V, distance_matrix, energy_matrix, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_model * 2)  # scores : [batch_size, n_heads, seq_len, seq_len]
        #print(scores[:l//2].device)
        # flag = distance_matrix[:l // 2]
        # print(flag.device)

        # scores_substrate = scores[:l//2] + self.adapter(self.w_distance_substrate * distance_matrix[:l//2])
        # scores_enzyme = scores[l//2:] + self.w_energy_enzyme * energy_matrix+ self.w_distance_enzyme * distance_matrix[l//2:]

        scores = scores.masked_fill(mask == 0, -1e9)

        attn = nn.Softmax(dim=-1)(scores)

        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, gaussian_params):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model * 2 * n_heads)  # dk
        self.W_K = nn.Linear(d_model, d_model * 2 * n_heads)  # dk
        self.W_V = nn.Linear(d_model, d_model * 2 * n_heads)  # dv
        self.d_model = d_model
        self.n_heads = n_heads
        self.liner = nn.Linear(n_heads * self.d_model * 2, self.d_model)
        self.layernorm = nn.LayerNorm(self.d_model)
        self.gaussian_params = gaussian_params

        #self.attention = ScaledDotProductAttention(self.d_model)
        #self.attention_gauss = ScaledDotProductAttention_Gauss(self.d_model, gaussian_params, K=gaussian_params.K)

    def forward(self, Q, K, V, distance_matrix, energy_matrix, mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_model * 2).transpose(1,
                                                                                         2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_model * 2).transpose(1,
                                                                                         2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_model * 2).transpose(1,
                                                                                         2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        #context = self.attention(q_s, k_s, v_s, distance_matrix, energy_matrix, mask)
        #context = self.attention_gauss(q_s, k_s, v_s, distance_matrix, energy_matrix, mask)
        #context = ScaledDotProductAttention(self.d_model)(q_s, k_s, v_s, distance_matrix, energy_matrix, mask)
        context = ScaledDotProductAttention_Gauss(self.d_model, self.gaussian_params, self.gaussian_params.K)(q_s, k_s, v_s, distance_matrix, energy_matrix, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.n_heads * self.d_model * 2)  # context: [batch_size, seq_len, n_heads, d_v]
        output = self.liner(context)
        return self.layernorm(output + residual)  # output: [batch_size, seq_len, d_model]




class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, gaussian_params):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, gaussian_params=gaussian_params)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model)

    def forward(self, input_feature, distance_matrix, energy_matrix, mask):
        enc_outputs = self.enc_self_attn(input_feature, input_feature, input_feature, distance_matrix, energy_matrix,
                                         mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class Transformer_DE(nn.Module):
    def __init__(self, n_layers=1, d_model=480, n_heads=2,K=10, device="cuda"):
        super(Transformer_DE, self).__init__()

        self.K = K
        self.gaussian_params = GaussianParameters(K=K)

        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_heads=n_heads, gaussian_params=self.gaussian_params) for _ in range(n_layers)])
        self.layers_substrate = nn.ModuleList([EncoderLayer(d_model=d_model, n_heads=n_heads, gaussian_params=self.gaussian_params) for _ in range(n_layers)])
        #self.Crossmodel = CrossAttentionWithWeight(d_model)
        self.Pool_Cat_MLP = Pool_Cat_MLP(d_model,self.gaussian_params)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Dropout(p=0.3),
            nn.SELU(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.3),
            nn.SELU(),
            nn.Linear(64, 1)
        )

    #def forward(self, input_feature, distance_matrix, energy_matrix, mask):
    def forward(self, Substrate_features, Substrate_distances, Substrate_mask, Enzymes_features, Enzymes_distances, Enzymes_energy, Enzymes_mask):

        for layer in self.layers:
            Enzymes_features = layer(Enzymes_features, Enzymes_distances, Enzymes_energy, Enzymes_mask)

        for layer in self.layers_substrate:
            Substrate_features = layer(Substrate_features, Substrate_distances, None,Substrate_mask)

        Activate_Site_f = self.classifier(Enzymes_features)  # [batch_size, 2] predict isNext

        Activate_Site = torch.sigmoid(Activate_Site_f)

        # Q = output_substrate
        # V = K = output_enzyme

        #Q = esm_features_substrate_batch

        #Cleavage_Site, _ = self.Crossmodel(Q, K, V, Activate_Site)
        Cleavage_Site = self.Pool_Cat_MLP(Enzymes_features, Substrate_features, Activate_Site,Activate_Site_f,Enzymes_mask)

        return Activate_Site, Cleavage_Site


# Load data
# Enzyme = os.listdir("/home/shuoyan/Chenao/FRUS/Enzyme/pdb_files")
with open("../Data_Split/enzyme_dict_1500_3t1.pkl", "rb") as f:
    enzyme_dict = pickle.load(f)

# new_enzyme_dict={}
# new_enzyme_dict['MER0028414']=enzyme_dict['MER0028414']
# new_enzyme_dict['MER0004552']=enzyme_dict['MER0004552']
#
# enzyme_dict=new_enzyme_dict

# Substrate = os.listdir("/mnt/data2/Chenao/substrate_alphafold_structures")
substrate_dict = pickle.load(open("../Data_Split/substrate_uniprot_seq_1500_3t1.pkl", "rb"))


# Load ESM-2 model
esmmodel, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
esmmodel = nn.DataParallel(esmmodel, list(range(0,8)))
batch_converter = alphabet.get_batch_converter()
esmmodel.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
esmmodel = esmmodel.to(device)

Transformer_DE = Transformer_DE(d_model=480, n_heads=num_attention_heads, n_layers=num_transformer_layers,K=K).to(device)

state = torch.load("model_weightes/UniZyme.pt")
new_state_dict = {}
for k, v in state.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
Transformer_DE.load_state_dict(new_state_dict, strict=False)
del new_state_dict
del state

# state = torch.load("Transformer_DE.pt0.10997684611558164")
# new_state_dict = {}
# for k, v in state.items():
#     name = k[7:]  # remove `module.`
#     new_state_dict[name] = v
# Transformer_DE.load_state_dict(new_state_dict)



Transformer_DE = nn.DataParallel(Transformer_DE, list(range(0,8)))

zeroshot_A01009 = pickle.load(open("../data/test_A01009_zeroshot.pkl", "rb"))
zeroshot_M10004 = pickle.load(open("../data/test_M10004_zeroshot.pkl", "rb"))
test_C14005 = pickle.load(open("../data/test_C14005.pkl", "rb"))
test_C14003 = pickle.load(open("../data/test_C14003.pkl", "rb"))
test_M10003 = pickle.load(open("../data/test_M10003.pkl", "rb"))

#结合
validation_dataset={}
validation_dataset.update(zeroshot_A01009)
validation_dataset.update(zeroshot_M10004)
validation_dataset.update(test_C14005)
validation_dataset.update(test_C14003)
validation_dataset.update(test_M10003)


zeroshot_A01009_l = [[k, v] for k, v in zeroshot_A01009.items()]
zeroshot_M10004_l = [[k, v] for k, v in zeroshot_M10004.items()]
test_C14005_l = [[k, v] for k, v in test_C14005.items()]
test_C14003_l = [[k, v] for k, v in test_C14003.items()]
test_M10003_l = [[k, v] for k, v in test_M10003.items()]

validation_data_A01009 = SequenceDataset(zeroshot_A01009_l, substrate_dict, enzyme_dict)
validation_data_M10004 = SequenceDataset(zeroshot_M10004_l, substrate_dict, enzyme_dict)
validation_data_C14005 = SequenceDataset(test_C14005_l, substrate_dict, enzyme_dict)
validation_data_C14003 = SequenceDataset(test_C14003_l, substrate_dict, enzyme_dict)
validation_data_M10003 = SequenceDataset(test_M10003_l, substrate_dict, enzyme_dict)

zeroshot_loader_A01009 = DataLoader(validation_data_A01009, batch_size=real_batch_size//2, shuffle=False, num_workers=8)
zeroshot_loader_M10004 = DataLoader(validation_data_M10004, batch_size=real_batch_size//2, shuffle=False, num_workers=8)
validation_loader_C14005 = DataLoader(validation_data_C14005, batch_size=real_batch_size//2, shuffle=False, num_workers=8)
validation_loader_C14003 = DataLoader(validation_data_C14003, batch_size=real_batch_size//2, shuffle=False, num_workers=8)
validation_loader_M10003 = DataLoader(validation_data_M10003, batch_size=real_batch_size//2, shuffle=False, num_workers=8)


validation_loaders={"validation_loader_A01009":zeroshot_loader_A01009,"zeroshot_loader_M10004":zeroshot_loader_M10004,"validation_loader_C14005":validation_loader_C14005,"validation_loader_C14003":validation_loader_C14003,"validation_loader_M10003":validation_loader_M10003}
validation_loaders_keys=validation_loaders.keys()

Transformer_DE.eval()
with torch.no_grad():
    for eachloader_name in validation_loaders_keys:
        eachloader=validation_loaders[eachloader_name]
        validation_loss = 0.0
        output_true = []
        output_pred = []

        for i, (batch_tokens_uniprot, batch_tokens_enzyme, batch_lens_uniprot, batch_lens_enzyme, Enzymes_distances,
                Substrate_distances, Enzymes_energy, Enzymes_mask, Substrate_mask, activate_labels_batch,
                cleavage_labels_batch) in enumerate(eachloader):

            batch_tokens = torch.cat((batch_tokens_uniprot, batch_tokens_enzyme), dim=0).to(device)
            batch_lens = torch.cat((batch_lens_uniprot, batch_lens_enzyme), dim=0).to(device)
            Enzymes_distances = Enzymes_distances.to(device)
            Substrate_distances = Substrate_distances.to(device)
            Enzymes_energy = Enzymes_energy.to(device)
            Enzymes_mask = Enzymes_mask.to(device)
            Substrate_mask = Substrate_mask.to(device)
            activate_labels_batch = activate_labels_batch.to(device)
            cleavage_labels_batch = cleavage_labels_batch.to(device)

            batch_size = batch_tokens.size(0) // 2
            if batch_size == 0:
                continue

            print(str(i) + "/" + str(len(eachloader)))
            print(batch_size)

            with torch.no_grad():
                try:
                    result = esmmodel(batch_tokens, repr_layers=[12], return_contacts=False)
                    token_representations = result["representations"][12]
                except:
                    continue

            padded_esm_features = []

            max_len = 1500

            for i, seq_len in enumerate(batch_lens):
                esm_features = token_representations[i][1:seq_len + 1, :]

                # 填充 esm_features
                pad_size = max_len - seq_len
                if pad_size > 0:
                    pad_esm = torch.zeros(pad_size, esm_features.size(1)).to(device)
                    esm_features_padded = torch.cat([esm_features, pad_esm], dim=0)  # (max_len, d_model)
                else:
                    esm_features_padded = esm_features[:max_len, :].to(device)

                padded_esm_features.append(esm_features_padded)

            # 使用success_real对数据进行二次筛选
            padded_esm_features_real = []

            for each in range(batch_size):
                # each对应uniprot，each+batch_size对应mernum
                padded_esm_features_real.append(padded_esm_features[each])

            for each in range(batch_size):
                padded_esm_features_real.append(padded_esm_features[each + batch_size])

            # esm_features_batch = torch.stack(padded_esm_features).to(device)  # (batch_size, max_len, d_model)
            # distances_batch = torch.stack(padded_distances).to(device)  # (batch_size, max_len, max_len)
            # energy_batch = torch.stack(padded_energy).to(device)  # (batch_size, max_len, max_len)
            # activate_labels_batch = torch.stack(activate_padded_labels).to(device)  # (batch_size, max_len)
            # cleavage_labels_batch = torch.stack(padded_cleavage_labels_real).to(device)  # (batch_size, max_len

            esm_features_batch = torch.stack(padded_esm_features_real)  # (batch_size, max_len, d_model)

            size = esm_features_batch.size(0)
            print(size)

            Enzymes_features = esm_features_batch[size // 2:].to(device)
            Substrate_features = esm_features_batch[:size // 2].to(device)

            Activate_Site, Cleavage_Site = Transformer_DE(Substrate_features, Substrate_distances, Substrate_mask,
                                                          Enzymes_features, Enzymes_distances, Enzymes_energy,
                                                          Enzymes_mask)# Activate_Site, Cleavage_Site = Transformer_DE(esm_features_batch, distances_batch, energy_batch, mask)

            Activate_Site = Activate_Site.view(-1, max_len)
            Cleavage_Site = Cleavage_Site.view(-1, max_len)

            activate_labels_batch = activate_labels_batch.view(-1, max_len)
            cleavage_labels_batch = cleavage_labels_batch.view(-1, max_len)


            # 酶位点的loss
            loss_active = F.binary_cross_entropy(Activate_Site, activate_labels_batch, reduction="none")
            loss_cleavage = F.binary_cross_entropy(Cleavage_Site, cleavage_labels_batch, reduction="none")

            Enzymes_mask = Enzymes_mask[:, 0, :, 0]
            weighted_Enzymes_mask = activate_labels_batch * 9 + Enzymes_mask
            loss_active = loss_active * weighted_Enzymes_mask
            loss_active = loss_active.sum() / (Enzymes_mask.sum() + activate_labels_batch.sum()*9)

            # 打印labels_batch为1的output
            print("Activate_Site")
            print(Activate_Site)
            print(Activate_Site[activate_labels_batch == 1])

            Substrate_mask = Substrate_mask[:, 0, :, 0]
            weighted_Substrate_mask = cleavage_labels_batch * 9 + Substrate_mask
            loss_cleavage = loss_cleavage * weighted_Substrate_mask
            loss_cleavage = loss_cleavage.sum() / (Substrate_mask.sum() + cleavage_labels_batch.sum()*9)

            # 打印labels_batch为1的output
            print("Cleavage_Site")
            print(Cleavage_Site)
            print(Cleavage_Site[cleavage_labels_batch == 1])

            eachloaderloss = loss_cleavage + 10*loss_active  # +loss_active

            Cleavage_Site = Cleavage_Site.cpu().detach().numpy()
            cleavage_labels_batch = cleavage_labels_batch.cpu().detach().numpy()

            for i in range(size // 2):
                length = batch_lens_uniprot[i]
                output_true.append(cleavage_labels_batch[i][:length])
                output_pred.append(Cleavage_Site[i][:length])

            validation_loss += eachloaderloss.item()

        #存储output_true、output_pred
        with open("UniZyme"+eachloader_name+"output_true.pkl","wb") as f:
            pickle.dump(output_true,f)
        with open("UniZyme"+eachloader_name+"output_pred.pkl","wb") as f:
            pickle.dump(output_pred,f)