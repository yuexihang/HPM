import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from model.Embedding import timestep_embedding
from model.spectral_embedding import robust_spectral
import math
from einops import rearrange

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Calibrated_Spectral_Mixer_Temporal(nn.Module):
    """For irregular meshes with temporal dimension."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., freq_num=64,
                 freq_num_time=16, spectral_trans_time_length=121,
                 spectral_embedding=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1, 1]) * 0.5)
        self.temperature_time = nn.Parameter(torch.ones([1, heads, 1, 1, 1]) * 0.5)

        # Mixer Modules
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)

        self.mlp_trans_weights = nn.Parameter(torch.empty((dim_head, dim_head)))
        torch.nn.init.kaiming_uniform_(self.mlp_trans_weights, a=math.sqrt(5))

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # Spatial gates
        self.in_project_gates = nn.Linear(dim_head, freq_num)
        for l in [self.in_project_gates]:
            torch.nn.init.orthogonal_(l.weight)

        # Temporal gates
        self.in_project_gates_time = nn.Linear(dim_head, freq_num_time)
        for l in [self.in_project_gates_time]:
            torch.nn.init.orthogonal_(l.weight)

        self.time_process = nn.Linear(dim_head, dim_head)

        # Temporal Eigenfunctions
        fourier_matrix_time = robust_spectral.onedim_spectral_meshes(spectral_trans_time_length, freq_num_time)  # T, G_time
        self.fourier_matrix_time = nn.Parameter(fourier_matrix_time, requires_grad=False)
        self.layernorm_time = nn.LayerNorm((freq_num_time, dim_head))

        # Spatial Eigenfunctions (fixed from mesh Laplacian)
        self.layernorm2 = nn.LayerNorm((freq_num, dim_head))

        fixed_spectral_embedding = torch.from_numpy(spectral_embedding).float()
        spectral_emb = fixed_spectral_embedding[:, :freq_num]                   # N, freq_num
        spectral_emb = spectral_emb[None, :, :].repeat(heads, 1, 1)            # H, N, freq_num
        spectral_emb = F.normalize(spectral_emb, p=2, dim=-1)                  # L2 Norm
        self.inver = nn.Parameter(spectral_emb, requires_grad=False)            # H, N, freq_num

    def forward(self, x):
        B, N, T, C = x.shape
        spectral_embedding = self.inver[None, :, :, :]                          # 1, H, N, M

        fx_mid = self.in_project_fx(x).reshape(B, N, T, self.heads, self.dim_head).permute(0, 3, 1, 2, 4).contiguous()  # B H N T C
        x_mid = self.in_project_x(x).reshape(B, N, T, self.heads, self.dim_head).permute(0, 3, 1, 2, 4).contiguous()    # B H N T C

        ### Spatial Spectral Transform (gated)
        eigen_gate = self.softmax(self.in_project_gates(x_mid) / self.temperature)  # B H N T G
        eigens = eigen_gate * spectral_embedding[:, :, :, None, :]              # B H N T G
        spectral_feature = torch.einsum("bhntc,bhntg->bhtgc", fx_mid, eigens)

        # Spatial Spectral Domain Processing
        bsize, hsize, tsize, gsize, csize = spectral_feature.shape
        spectral_feature = self.layernorm2(spectral_feature.reshape(-1, gsize, csize)).reshape(bsize, hsize, tsize, gsize, csize)
        out_spectral_feature = torch.einsum("bhtgi,io->bhtgo", spectral_feature, self.mlp_trans_weights)

        # Spatial Inverse Spectral Transform
        fx_mid = torch.einsum("bhtgc,bhntg->bhntc", out_spectral_feature, eigens)  # B H N T C

        ### Temporal Spectral Transform (gated)
        eigen_gate_time = self.softmax(self.in_project_gates_time(x_mid) / self.temperature_time)   # B H N T G_time
        eigens_time = eigen_gate_time * self.fourier_matrix_time[None, None, None, :, :]             # B H N T G_time
        spectral_feature_time = torch.einsum("bhntc,bhntg->bhngc", fx_mid, eigens_time)

        # Temporal Spectral Domain Processing
        bsize, hsize, nsize, gsize, csize = spectral_feature_time.shape
        spectral_feature_time = self.layernorm_time(spectral_feature_time.reshape(-1, gsize, csize)).reshape(bsize, hsize, nsize, gsize, csize)
        out_spectral_feature_time = self.time_process(spectral_feature_time)     # B H N G C

        # Temporal Inverse Spectral Transform
        out_x = torch.einsum("bhngc,bhntg->bhntc", out_spectral_feature_time, eigens_time)

        out_x = rearrange(out_x, 'b h n t d -> b n t (h d)')
        return self.to_out(out_x)


class Mixer_Block(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            freq_num=64,
            freq_num_time=16,
            spectral_trans_time_length=121,
            spectral_embedding=None,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Calibrated_Spectral_Mixer_Temporal(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                       dropout=dropout, freq_num=freq_num,
                                                       freq_num_time=freq_num_time,
                                                       spectral_trans_time_length=spectral_trans_time_length,
                                                       spectral_embedding=spectral_embedding)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self,
                 space_dim=0,
                 n_layers=4,
                 n_hidden=64,
                 dropout=0.0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=6,
                 out_dim=3,
                 freq_num=64,
                 ref=8,
                 unified_pos=False,
                 spectral_pos_embedding=32,
                 freq_num_time=16,
                 spectral_trans_time_length=121,
                 spectral_embedding=None,
                 ):
        super(Model, self).__init__()
        self.__name__ = 'HPM_Irregular_Temporal'
        self.n_hidden = n_hidden
        self.spectral_pos_embedding = spectral_pos_embedding

        self.preprocess = MLP(fun_dim + space_dim + spectral_pos_embedding, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        # Fixed spectral embedding from mesh Laplacian eigenvectors
        fixed_spectral_embedding = torch.from_numpy(spectral_embedding).float()[None, :, :]  # 1, N, K
        self.fixed_spectral_embedding = nn.Parameter(fixed_spectral_embedding, requires_grad=False)

        self.blocks = nn.ModuleList([Mixer_Block(num_heads=n_head, hidden_dim=n_hidden,
                                                 dropout=dropout,
                                                 act=act,
                                                 mlp_ratio=mlp_ratio,
                                                 out_dim=out_dim,
                                                 freq_num=freq_num,
                                                 freq_num_time=freq_num_time,
                                                 spectral_trans_time_length=spectral_trans_time_length,
                                                 spectral_embedding=spectral_embedding,
                                                 last_layer=(lid == n_layers - 1))
                                     for lid in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx=None):
        B, T, _ = x.shape
        N = self.fixed_spectral_embedding.shape[1]

        x = x[:, None, :, :].repeat(1, N, 1, 1)  # B, N, T, fun_dim

        # Spectral Positional Embedding
        spectral_pos = self.fixed_spectral_embedding[..., :self.spectral_pos_embedding]  # 1, N, pos_emb
        x = torch.concat((x, spectral_pos[:, :, None, :].repeat(B, 1, T, 1)), dim=-1)  # B N T (fun_dim + pos_emb)

        fx = self.preprocess(x)  # B N T C

        # Time Embedding
        time_embedding = torch.arange(0, T, 1, device=x.device)  # T
        Time_emb = timestep_embedding(time_embedding, self.n_hidden)  # T, C
        Time_emb = self.time_fc(Time_emb)  # T, C
        fx = fx + Time_emb[None, None, :, :]  # B N T C

        for block in self.blocks:
            fx = block(fx)

        return fx
