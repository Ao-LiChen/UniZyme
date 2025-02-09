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

#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,0,2,3,4,5,6,7"

# Parameters
distance_threshold = 8.0
num_epochs = 10
learning_rate = 0.0001
real_batch_size = 48
train_ratio = 0.8  # 80% of data for training
num_attention_heads = 2
num_transformer_layers = 1
K=10


class SequenceDataset(Dataset):
    def __init__(self, data, uniprot_dict, mernum_dict):
        self.data = data
        self.uniprot_dict = uniprot_dict
        self.mernum_dict = mernum_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        key = entry[0]

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
                    energy = pickle.load(open(f"../data/Enzyme_Energy/{batch_labels[i]}_configurational.pkl", "rb"))
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

            distances_padded = distances_padded.unsqueeze(0)
            distances_padded = distances_padded.expand(num_attention_heads, -1, -1)

            energy_padded = energy_padded.unsqueeze(0)
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


        mask = torch.zeros(len(distances_batch), num_attention_heads, max_len, max_len)

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
    distances = 1 / (distances + 1)
    distances = torch.tensor(distances, dtype=torch.float)
    # distances[distances > distance_threshold] = 0  # Apply threshold
    return distances


class GaussianParameters(nn.Module):
    def __init__(self, K=10):
        super(GaussianParameters, self).__init__()
        self.K = K
        self.mu_D_Enzyme = nn.Parameter(torch.randn(K))        # Distance Gaussian Kernel Center
        self.sigma_D_Enzyme = nn.Parameter(torch.ones(K))      # Distance Gaussian Kernel Standard Deviation
        self.b_D_Enzyme = nn.Parameter(torch.zeros(K))         # Distance Bias Term

        self.mu_D_Sub = nn.Parameter(torch.randn(K))        # Distance Gaussian Kernel Center
        self.sigma_D_Sub = nn.Parameter(torch.ones(K))      # Distance Gaussian Kernel Standard Deviation
        self.b_D_Sub = nn.Parameter(torch.zeros(K))         # Distance Bias Term

        self.mu_E = nn.Parameter(torch.randn(K))        # Energy Gaussian Kernel Center
        self.sigma_E = nn.Parameter(torch.ones(K))      # Energy Gaussian Kernel Standard Deviation
        self.b_E = nn.Parameter(torch.zeros(K))         # Energy Bias Term


        self.W_D1 = nn.Linear(K, K)  # [K x K]
        self.W_D2 = nn.Linear(K, 1)  # [K x 1]

        self.mu_activation = nn.Parameter(torch.randn(K))        # Activation Site Gaussian Kernel Center
        self.sigma_activation = nn.Parameter(torch.ones(K))      # Activation Site Gaussian Kernel Standard Deviation
        self.b_activation = nn.Parameter(torch.zeros(K))         # Activation Site Bias Term

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
        #self.conv1d = nn.Conv1d(in_channels=2 * d, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1d = nn.Conv1d(in_channels=2*d, out_channels=128, kernel_size=31, stride=1, padding=15)
        self.conv_activation = nn.SELU()
        self.conv_dropout = nn.Dropout(p=0.2)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.gaussian_params = gaussian_params

    def forward(self, Enzyme, Substrate, Activate_Site,Activate_Site_f,Enzymes_mask):


        Enzymes_mask = Enzymes_mask[:, 0, :, 0]

        Enzymes_length = Enzymes_mask.sum(dim=-1)
        Enzymes_mask = Enzymes_mask.unsqueeze(-1).repeat(1, 1, self.d)


        Enzyme = Enzyme * Enzymes_mask

        pooled_Enzyme = Enzyme.sum(dim=-2) / Enzymes_length.view(-1, 1)


        pooled_Enzyme = pooled_Enzyme.unsqueeze(1).repeat(1, Substrate.size(1), 1)


        combined_features = torch.cat((Substrate,pooled_Enzyme), dim=-1)
        # prediction = self.mlp(combined_features)

        combined_features = combined_features.permute(0, 2, 1)


        conv_output = self.conv1d(combined_features)  # (batch, 64, length_sub)
        conv_output = self.conv_activation(conv_output)
        conv_output = self.conv_dropout(conv_output)


        conv_output = conv_output.permute(0, 2, 1)


        prediction = self.fc(conv_output)  # (batch, length_sub, 1)

        return prediction





# Define the Transformer model with multi-layer attention
class ScaledDotProductAttention_Gauss(nn.Module):
    def __init__(self, d_model,gaussian_params, K=10):
        super(ScaledDotProductAttention_Gauss, self).__init__()
        self.d_model = d_model
        self.K = K # Number of Gaussian Kernels

        self.gaussian_params = gaussian_params



    def forward(self, Q, K, V, distance_matrix, energy_matrix, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_model * 2)  # scores : [batch_size, n_heads, seq_len, seq_len]
        l= scores.size(0)
        batch_size, n_heads, seq_len, _ = scores.shape

        distance_matrix = distance_matrix.unsqueeze(-1).repeat(1,1, 1, 1, self.K)
        #print(distance_matrix.shape)

        if energy_matrix is None:

            distance = distance_matrix + self.gaussian_params.b_D_Sub  # [batch_sub, n_heads, seq, seq]

            # [batch_sub, n_heads, seq, seq, K]
            psi_dist = torch.exp(-0.5 * ((distance - self.gaussian_params.mu_D_Sub) / self.gaussian_params.sigma_D_Sub) ** 2)
            psi_dist = psi_dist / (
                        torch.sqrt(torch.tensor(2 * math.pi, device=distance.device)) * self.gaussian_params.sigma_D_Sub)

            psi_dist_flat = psi_dist
            psi_dist_flat = F.gelu(self.gaussian_params.W_D1(psi_dist_flat))  # [B*H*S*S, K]
            psi_dist_flat = self.gaussian_params.W_D2(psi_dist_flat)  # [B*H*S*S, 1]
            #Phi_dist_substrate = Phi_dist_substrate_flat.view(l//2, n_heads, seq_len, seq_len)  # [batch_sub, n_heads, seq, seq]

            Phi_dist = psi_dist_flat.squeeze(-1)  # [batch_sub, n_heads, seq, seq]



            #Phi_dist = self.adapter(Phi_dist)

            scores = scores + Phi_dist  # [batch_sub, n_heads, seq, seq]
        else:

            distance = distance_matrix + self.gaussian_params.b_D_Enzyme  # [batch_sub, n_heads, seq, seq]

            # [batch_sub, n_heads, seq, seq, K]
            psi_dist = torch.exp(
                -0.5 * ((distance - self.gaussian_params.mu_D_Enzyme) / self.gaussian_params.sigma_D_Enzyme) ** 2)
            psi_dist = psi_dist / (
                    torch.sqrt(torch.tensor(2 * math.pi, device=distance.device)) * self.gaussian_params.sigma_D_Enzyme)

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


            # psi_energy_enzyme_flat = psi_energy_enzyme.view(-1, self.K)  # [B*H*S*S, K]
            psi_energy_flat = psi_energy
            psi_energy_flat = F.gelu(self.gaussian_params.W_D1(psi_energy_flat))  # [B*H*S*S, K]
            psi_energy_flat = self.gaussian_params.W_D2(psi_energy_flat)  # [B*H*S*S, 1]
            # Phi_energy_enzyme = Phi_energy_enzyme_flat.view(l//2, n_heads, seq_len, seq_len)  # [batch_enzyme, n_heads, seq, seq]
            Phi_energy = psi_energy_flat.squeeze(-1)  # [batch_enzyme, n_heads, seq, seq]


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


    def forward(self, Q, K, V, distance_matrix, energy_matrix, mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_model * 2)  # scores : [batch_size, n_heads, seq_len, seq_len]


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


        context = ScaledDotProductAttention(self.d_model)(q_s, k_s, v_s, distance_matrix, energy_matrix, mask)
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
        #self.gaussian_params = GaussianParameters(K=K)
        self.gaussian_params =None
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

        self.reactzyme_mlp = nn.Linear(d_model, d_model)

    #def forward(self, input_feature, distance_matrix, energy_matrix, mask):
    def forward(self, Substrate_features, Substrate_distances, Substrate_mask, Enzymes_features, Enzymes_distances, Enzymes_energy, Enzymes_mask):

        # for layer in self.layers:
        #     Enzymes_features = layer(Enzymes_features, Enzymes_distances, Enzymes_energy, Enzymes_mask)

        Enzymes_features = self.reactzyme_mlp(Enzymes_features)

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


with open("../Data_Split/enzyme_dict_1500_3t1.pkl", "rb") as f:
    enzyme_dict = pickle.load(f)


substrate_dict = pickle.load(open("../Data_Split/substrate_uniprot_seq_1500_3t1.pkl", "rb"))


# Load ESM-2 model
esmmodel, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
esmmodel = nn.DataParallel(esmmodel, list(range(0,8)))
batch_converter = alphabet.get_batch_converter()
esmmodel.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
esmmodel = esmmodel.to(device)

Transformer_DE = Transformer_DE(d_model=480, n_heads=num_attention_heads, n_layers=num_transformer_layers,K=K).to(device)


Transformer_DE = nn.DataParallel(Transformer_DE, list(range(0,8)))


optimizer = torch.optim.Adam(Transformer_DE.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=1, verbose=True)
early_stopping = EarlyStopping(patience=3, verbose=True)

train_dataset = pickle.load(open("../Data_Split/train.pkl", "rb"))

train_dataset_l = [[k, v] for k, v in train_dataset.items()]

test_dataset = pickle.load(open("../Data_Split/test.pkl", "rb"))
test_dataset_l = [[k, v] for k, v in test_dataset.items()]
#test_dataset_l = [[k, v] for k, v in test_dataset.items() if random.random() < 0.0003]

train_data = SequenceDataset(train_dataset_l, substrate_dict, enzyme_dict)
test_data = SequenceDataset(test_dataset_l, substrate_dict, enzyme_dict)

train_loader = DataLoader(train_data, batch_size=real_batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, batch_size=real_batch_size, shuffle=False, num_workers=8)


# Training loop
train_losses, test_losses = [], []
#
best_test_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = 0.0
    Transformer_DE.train()
    for i, (batch_tokens_uniprot,batch_tokens_enzyme,batch_lens_uniprot,batch_lens_enzyme,Enzymes_distances,Substrate_distances,Enzymes_energy,Enzymes_mask,Substrate_mask,activate_labels_batch,cleavage_labels_batch) in enumerate(train_loader):

        batch_tokens = torch.cat((batch_tokens_uniprot,batch_tokens_enzyme),dim=0).to(device)
        batch_lens=torch.cat((batch_lens_uniprot,batch_lens_enzyme),dim=0).to(device)
        Enzymes_distances = Enzymes_distances.to(device)
        Substrate_distances = Substrate_distances.to(device)
        Enzymes_energy = Enzymes_energy.to(device)
        Enzymes_mask = Enzymes_mask.to(device)
        Substrate_mask = Substrate_mask.to(device)
        activate_labels_batch = activate_labels_batch.to(device)
        cleavage_labels_batch = cleavage_labels_batch.to(device)

        batch_size = batch_tokens.size(0)//2
        if batch_size == 0:
            continue


        print(str(i) + "/" + str(len(train_loader)))
        print(batch_size)

        with torch.no_grad():
            try:
                result = esmmodel(batch_tokens, repr_layers=[12], return_contacts=False)
                token_representations = result["representations"][12]
            except Exception as e:
                print(e)
                continue

        padded_esm_features = []

        max_len = 1500

        for i, seq_len in enumerate(batch_lens):
            esm_features = token_representations[i][1:seq_len + 1, :]

            # padding esm_features
            pad_size = max_len - seq_len
            if pad_size > 0:
                pad_esm = torch.zeros(pad_size, esm_features.size(1)).to(device)
                esm_features_padded = torch.cat([esm_features, pad_esm], dim=0)  # (max_len, d_model)
            else:
                esm_features_padded = esm_features[:max_len, :].to(device)

            padded_esm_features.append(esm_features_padded)


        padded_esm_features_real = []

        for each in range(batch_size):
            padded_esm_features_real.append(padded_esm_features[each])


        for each in range(batch_size):
            padded_esm_features_real.append(padded_esm_features[each + batch_size])

        esm_features_batch = torch.stack(padded_esm_features_real) # (batch_size, max_len, d_model)


        size=esm_features_batch.size(0)
        print(size)

        Enzymes_features = esm_features_batch[size//2:].to(device)
        Substrate_features = esm_features_batch[:size//2].to(device)




        Activate_Site, Cleavage_Site = Transformer_DE(Substrate_features, Substrate_distances, Substrate_mask, Enzymes_features, Enzymes_distances, Enzymes_energy, Enzymes_mask)


        optimizer.zero_grad()

        Activate_Site=Activate_Site.view(-1, max_len)
        Cleavage_Site=Cleavage_Site.view(-1, max_len)

        activate_labels_batch = activate_labels_batch.view(-1, max_len)
        cleavage_labels_batch = cleavage_labels_batch.view(-1, max_len)

        loss_active = F.binary_cross_entropy(Activate_Site, activate_labels_batch, reduction="none")
        loss_cleavage = F.binary_cross_entropy(Cleavage_Site, cleavage_labels_batch, reduction="none")

        Enzymes_mask = Enzymes_mask[:, 0, :, 0]
        weighted_Enzymes_mask = activate_labels_batch * 9 + Enzymes_mask
        loss_active = loss_active * weighted_Enzymes_mask
        loss_active = loss_active.sum() / (Enzymes_mask.sum() + activate_labels_batch.sum() * 9)

        print("Activate_Site")
        print(Activate_Site)
        print(Activate_Site[activate_labels_batch == 1])

        Substrate_mask = Substrate_mask[:, 0, :, 0]
        weighted_Substrate_mask = cleavage_labels_batch * 9 + Substrate_mask
        loss_cleavage = loss_cleavage * weighted_Substrate_mask
        loss_cleavage = loss_cleavage.sum() / (Substrate_mask.sum() + cleavage_labels_batch.sum() * 9)

        print("Cleavage_Site")
        print(Cleavage_Site)
        print(Cleavage_Site[cleavage_labels_batch == 1])


        loss = loss_cleavage #+ loss_active # +loss_active
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        #print(loss.item())
        print("train_loss", loss.item(), "train_loss_active", loss_active.item(), "train_loss_cleavage", loss_cleavage.item())

    print(f"Epoch {epoch + 1}/{num_epochs}, Batch Loss: {train_loss / len(train_loader)}")

    train_losses.append(train_loss / len(train_loader))


    # Evaluation
    test_loss = 0.0
    Transformer_DE.eval()
    with torch.no_grad():
        for i, (batch_tokens_uniprot, batch_tokens_enzyme, batch_lens_uniprot, batch_lens_enzyme, Enzymes_distances,
                Substrate_distances, Enzymes_energy, Enzymes_mask, Substrate_mask, activate_labels_batch,
                cleavage_labels_batch) in enumerate(test_loader):

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

            print(str(i) + "/" + str(len(test_loader)))
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


                pad_size = max_len - seq_len
                if pad_size > 0:
                    pad_esm = torch.zeros(pad_size, esm_features.size(1)).to(device)
                    esm_features_padded = torch.cat([esm_features, pad_esm], dim=0)  # (max_len, d_model)
                else:
                    esm_features_padded = esm_features[:max_len, :].to(device)

                padded_esm_features.append(esm_features_padded)


            padded_esm_features_real = []

            for each in range(batch_size):
                padded_esm_features_real.append(padded_esm_features[each])

            for each in range(batch_size):
                padded_esm_features_real.append(padded_esm_features[each + batch_size])

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


            loss_active = F.binary_cross_entropy(Activate_Site, activate_labels_batch, reduction="none")
            loss_cleavage = F.binary_cross_entropy(Cleavage_Site, cleavage_labels_batch, reduction="none")

            Enzymes_mask = Enzymes_mask[:, 0, :, 0]
            weighted_Enzymes_mask = activate_labels_batch * 9 + Enzymes_mask
            loss_active = loss_active * weighted_Enzymes_mask
            loss_active = loss_active.sum() / (Enzymes_mask.sum() + activate_labels_batch.sum()*9)

            print("Activate_Site")
            print(Activate_Site)
            print(Activate_Site[activate_labels_batch == 1])

            Substrate_mask = Substrate_mask[:, 0, :, 0]
            weighted_Substrate_mask = cleavage_labels_batch * 9 + Substrate_mask
            loss_cleavage = loss_cleavage * weighted_Substrate_mask
            loss_cleavage = loss_cleavage.sum() / (Substrate_mask.sum() + cleavage_labels_batch.sum()*9)

            print("Cleavage_Site")
            print(Cleavage_Site)
            print(Cleavage_Site[cleavage_labels_batch == 1])

            loss = loss_cleavage #+ loss_active  # +loss_active



            test_loss += loss.item()
            #print loss
            print("test_loss",loss.item(),"test_loss_active",loss_active.item(),"test_loss_cleavage",loss_cleavage.item())


        print(f"Epoch {epoch + 1}/{num_epochs}, Batch Loss: {test_loss / len(test_loader)}")
        test_losses.append(test_loss / len(test_loader))


    scheduler.step(test_loss / len(test_loader))
    print(optimizer.param_groups[0]['lr'])
    early_stopping(test_loss / len(test_loader), Transformer_DE)


    if best_test_loss > test_loss / len(test_loader):
        best_test_loss = test_loss / len(test_loader)
        torch.save(Transformer_DE.state_dict(), "ReactZyme.pt" + str(best_test_loss)+"_"+str(epoch))

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}")
