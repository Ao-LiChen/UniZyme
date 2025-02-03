import torch
import esm
import numpy as np
from scipy.spatial import distance_matrix
from Bio.PDB import PDBParser
import pandas as pd
import frustratometer
import argparse
import sys
import os

# Configuration parameters
num_attention_heads = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 1500  # Consistent with model input size

# Load ESM model
esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model = esm_model.to(device).eval()

# Energy calculation configuration
calc_config = {
    'k_electrostatics': 0,
    'min_sequence_separation_rho': 12,
    'min_sequence_separation_contact': 12,
    'kind': 'configurational'
}

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


def calculate_energy_matrix(pdb_path):
    """Calculate configurational frustration using frustratometer"""
    try:
        structure = frustratometer.Structure(pdb_path)
        if len(structure.sequence) > max_len:
            raise ValueError(f"Sequence length exceeds {max_len}")

        awsem = frustratometer.AWSEM(
            structure,
            **{k: v for k, v in calc_config.items() if k != 'kind'}
        )
        return awsem.frustration(kind=calc_config['kind'])
    except Exception as e:
        print(f"Energy calculation failed: {str(e)}")
        return None


def predict(enzyme_pdb, substrate_pdb, enzyme_seq, substrate_seq, model):
    """Main prediction pipeline"""
    energy_matrix = calculate_energy_matrix(enzyme_pdb)
    if energy_matrix is None:
        return None

    enzyme_dist = create_distance(enzyme_pdb)
    substrate_dist = create_distance(substrate_pdb)

    # Process sequence features
    enzyme_data = [("enzyme", enzyme_seq)]
    substrate_data = [("substrate", substrate_seq)]

    _, _, enzyme_tokens = batch_converter(enzyme_data)
    _, _, substrate_tokens = batch_converter(substrate_data)

    def prepare_features(tokens, seq_len):
        """Extract and pad ESM features"""
        rep = esm_model(tokens.to(device), repr_layers=[12])["representations"][12][0, 1:seq_len + 1]
        return torch.nn.functional.pad(rep, (0, 0, 0, max_len - seq_len))

    enzyme_feat = prepare_features(enzyme_tokens, len(enzyme_seq))
    substrate_feat = prepare_features(substrate_tokens, len(substrate_seq))

    def create_mask(seq_len):
        """Generate attention mask for given sequence length"""
        mask = torch.zeros(1, num_attention_heads, max_len, max_len)
        mask[..., :seq_len, :seq_len] = 1.0
        return mask.to(device)

    with torch.no_grad():
        _, cleavage = model(
            substrate_feat.unsqueeze(0),
            substrate_dist.unsqueeze(0).to(device),
            create_mask(len(substrate_seq)),
            enzyme_feat.unsqueeze(0),
            enzyme_dist.unsqueeze(0).to(device),
            torch.tensor(energy_matrix, dtype=torch.float).unsqueeze(0).to(device),
            create_mask(len(enzyme_seq))
        )

    return cleavage.squeeze().cpu().numpy()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Predict substrate cleavage sites using UniZyme model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--enzyme_pdb', required=True,
                        help='Path to enzyme PDB file')
    parser.add_argument('--substrate_pdb', required=True,
                        help='Path to substrate PDB file')
    parser.add_argument('--enzyme_seq', required=True,
                        help='Full enzyme sequence in FASTA format (without header)')
    parser.add_argument('--substrate_seq', required=True,
                        help='Full substrate sequence in FASTA format (without header)')
    parser.add_argument('--model_path', default='model_weights/UniZyme.pt',
                        help='Path to trained model weights')
    parser.add_argument('--output', default='predictions.csv',
                        help='Output CSV file path')

    args = parser.parse_args()

    # Validate input files
    for f in [args.enzyme_pdb, args.substrate_pdb, args.model_path]:
        if not os.path.exists(f):
            sys.exit(f"Error: Input file {f} not found!")

    # Initialize model
    model = Transformer_DE(d_model=480, n_heads=num_attention_heads, n_layers=1, K=10).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Run prediction
    predictions = predict(
        args.enzyme_pdb,
        args.substrate_pdb,
        args.enzyme_seq.strip(),
        args.substrate_seq.strip(),
        model
    )

    # Save results
    if predictions is not None:
        pd.DataFrame({
            "Position": range(1, len(predictions) + 1),
            "Probability": predictions
        }).to_csv(args.output, index=False)
        print(f"Success! Results saved to {args.output}")
    else:
        print("Prediction failed due to processing errors")


if __name__ == "__main__":
    main()