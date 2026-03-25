import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from model.Embedding import timestep_embedding
import math
from einops import rearrange

from model.HPM_Irregular_Mesh import Calibrated_Spectral_Mixer, Mixer_Block, MLP, ACTIVATION


class Calibrated_Spectral_Mixer_TwoDomain(nn.Module):
    """Cross-domain spectral mixer: encodes with input-domain eigenfunctions, decodes with output-domain eigenfunctions."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., freq_num=32,
                 spectral_embedding_input=None, spectral_embedding_output=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads

        # Mixer Modules
        self.in_project_fx = nn.Linear(dim, inner_dim)

        self.mlp_trans_weights = nn.Parameter(torch.empty((dim_head, dim_head)))
        torch.nn.init.kaiming_uniform_(self.mlp_trans_weights, a=math.sqrt(5))

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.layernorm2 = nn.LayerNorm((freq_num, dim_head))

        # Input-domain Eigenfunctions
        fixed_spectral_input = torch.from_numpy(spectral_embedding_input).float()
        spectral_emb_in = fixed_spectral_input[:, :freq_num]                    # N, freq_num
        spectral_emb_in = spectral_emb_in[None, :, :].repeat(heads, 1, 1)      # H, N, freq_num
        spectral_emb_in = F.normalize(spectral_emb_in, p=2, dim=-1)            # L2 Norm
        self.inver_input = nn.Parameter(spectral_emb_in, requires_grad=False)   # H, N, freq_num

        # Output-domain Eigenfunctions
        fixed_spectral_output = torch.from_numpy(spectral_embedding_output).float()
        spectral_emb_out = fixed_spectral_output[:, :freq_num]                  # N, freq_num
        spectral_emb_out = spectral_emb_out[None, :, :].repeat(heads, 1, 1)    # H, N, freq_num
        spectral_emb_out = F.normalize(spectral_emb_out, p=2, dim=-1)          # L2 Norm
        self.inver_output = nn.Parameter(spectral_emb_out, requires_grad=False) # H, N, freq_num

    def forward(self, x):
        B, N, C = x.shape

        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N C

        # Spectral Transform (input domain)
        spectral_feature = torch.einsum("bhnc,hng->bhgc", fx_mid, self.inver_input)

        # Spectral Domain Processing
        bsize, hsize, gsize, csize = spectral_feature.shape
        spectral_feature = self.layernorm2(spectral_feature.reshape(-1, gsize, csize)).reshape(bsize, hsize, gsize, csize)
        out_spectral_feature = torch.einsum("bhgi,io->bhgo", spectral_feature, self.mlp_trans_weights)

        # Inverse Spectral Transform (output domain)
        out_x = torch.einsum("bhgc,hng->bhnc", out_spectral_feature, self.inver_output)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')

        return self.to_out(out_x)


class Mixer_Block_TwoDomain(nn.Module):
    """Cross-domain block: no residual connection, no MLP."""
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            freq_num=32,
            spectral_embedding_input=None,
            spectral_embedding_output=None,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Calibrated_Spectral_Mixer_TwoDomain(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, freq_num=freq_num,
                                                         spectral_embedding_input=spectral_embedding_input,
                                                         spectral_embedding_output=spectral_embedding_output)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx))
        return fx


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=4,
                 n_hidden=32,
                 dropout=0.0,
                 n_head=1,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 freq_num=128,
                 ref=8,
                 unified_pos=False,
                 spectral_pos_embedding=0,
                 domain_change_layer_idx=1,
                 spectral_embedding_input=None,
                 spectral_embedding_output=None,
                 ):
        super(Model, self).__init__()
        self.__name__ = 'HPM_Irregular_TwoDomain'
        self.ref = ref
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.spectral_pos_embedding = spectral_pos_embedding

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim + self.spectral_pos_embedding, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        # Fixed spectral embeddings from mesh Laplacian eigenvectors (two domains)
        fixed_spectral_input = torch.from_numpy(spectral_embedding_input).float()[None, :, :]    # 1, N, K
        fixed_spectral_output = torch.from_numpy(spectral_embedding_output).float()[None, :, :]  # 1, N, K
        self.fixed_spectral_embedding_input = nn.Parameter(fixed_spectral_input, requires_grad=False)
        self.fixed_spectral_embedding_output = nn.Parameter(fixed_spectral_output, requires_grad=False)

        # Build layers: input_domain → twodomain_transition → output_domain
        twodomain_layer = Mixer_Block_TwoDomain(
            num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
            freq_num=freq_num,
            spectral_embedding_input=spectral_embedding_input,
            spectral_embedding_output=spectral_embedding_output)

        input_domain_layers = [Mixer_Block(
            num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
            act=act, mlp_ratio=mlp_ratio, out_dim=out_dim, freq_num=freq_num,
            spectral_embedding=spectral_embedding_input,
            last_layer=(lid == n_layers - 1))
            for lid in range(0, domain_change_layer_idx)]

        output_domain_layers = [Mixer_Block(
            num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
            act=act, mlp_ratio=mlp_ratio, out_dim=out_dim, freq_num=freq_num,
            spectral_embedding=spectral_embedding_output,
            last_layer=(lid == n_layers - 1))
            for lid in range(domain_change_layer_idx + 1, n_layers)]

        self.blocks = nn.ModuleList(input_domain_layers + [twodomain_layer] + output_domain_layers)

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

    def get_grid(self, x, batchsize=1):
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda().reshape(batchsize, self.ref * self.ref, 2)

        pos = torch.sqrt(torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None):
        if self.unified_pos:
            x = self.get_grid(x, x.shape[0])

        if self.spectral_pos_embedding > 0:
            spectral_pos = self.fixed_spectral_embedding_input.repeat(x.shape[0], 1, 1)  # B, N, K
            x = torch.concat((x, spectral_pos[..., :self.spectral_pos_embedding]), dim=-1)

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)

        return fx
