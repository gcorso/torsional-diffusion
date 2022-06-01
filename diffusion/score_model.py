import math

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter
import numpy as np
from e3nn.nn import BatchNorm
import diffusion.torus as torus


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.ReLU(),
            nn.Linear(n_edge_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)

        return out


class TensorProductScoreModel(torch.nn.Module):
    def __init__(self, in_node_features=74, in_edge_features=4, sigma_embed_dim=32, sigma_min=0.01 * np.pi,
                 sigma_max=np.pi, sh_lmax=2, ns=32, nv=8, num_conv_layers=4, max_radius=5, radius_embed_dim=50,
                 scale_by_sigma=True, use_second_order_repr=True, batch_norm=True, residual=True
                 ):
        super(TensorProductScoreModel, self).__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.max_radius = max_radius
        self.radius_embed_dim = radius_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma

        self.node_embedding = nn.Sequential(
            nn.Linear(in_node_features + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(in_edge_features + sigma_embed_dim + radius_embed_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns)
        )
        self.distance_expansion = GaussianSmearing(0.0, max_radius, radius_embed_dim)
        conv_layers = []

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                residual=residual,
                batch_norm=batch_norm
            )
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.final_edge_embedding = nn.Sequential(
            nn.Linear(radius_embed_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns)
        )

        self.final_tp = o3.FullTensorProduct(self.sh_irreps, "2e")

        self.bond_conv = TensorProductConvLayer(
            in_irreps=self.conv_layers[-1].out_irreps,
            sh_irreps=self.final_tp.irreps_out,
            out_irreps=f'{ns}x0o',
            n_edge_features=3 * ns,
            residual=False,
            batch_norm=batch_norm
        )

        self.final_linear = nn.Sequential(
            nn.Linear(ns, ns, bias=False),
            nn.Tanh(),
            nn.Linear(ns, 1, bias=False)
        )

    def forward(self, data):
        node_attr, edge_index, edge_attr, edge_sh = self.build_conv_graph(data)
        src, dst = edge_index

        node_attr = self.node_embedding(node_attr)
        edge_attr = self.edge_embedding(edge_attr)

        for layer in self.conv_layers:
            edge_attr_ = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1)
            node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh, reduce='mean')

        bonds, edge_index, edge_attr, edge_sh = self.build_bond_conv_graph(data, node_attr)

        bond_vec = data.pos[bonds[1]] - data.pos[bonds[0]]
        bond_attr = node_attr[bonds[0]] + node_attr[bonds[1]]

        bonds_sh = o3.spherical_harmonics("2e", bond_vec, normalize=True, normalization='component')
        edge_sh = self.final_tp(edge_sh, bonds_sh[edge_index[0]])

        edge_attr = torch.cat([edge_attr, node_attr[edge_index[1], :self.ns], bond_attr[edge_index[0], :self.ns]], -1)
        out = self.bond_conv(node_attr, edge_index, edge_attr, edge_sh, out_nodes=data.edge_mask.sum(), reduce='mean')

        out = self.final_linear(out)

        data.edge_pred = out.squeeze()
        data.edge_sigma = data.node_sigma[data.edge_index[0]][data.edge_mask]

        if self.scale_by_sigma:
            score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
            score_norm = torch.tensor(score_norm, device=data.x.device)
            data.edge_pred = data.edge_pred * torch.sqrt(score_norm)
        return data

    def build_bond_conv_graph(self, data, node_attr):

        bonds = data.edge_index[:, data.edge_mask].long()
        bond_pos = (data.pos[bonds[0]] + data.pos[bonds[1]]) / 2
        bond_batch = data.batch[bonds[0]]
        edge_index = radius(data.pos, bond_pos, self.max_radius, batch_x=data.batch, batch_y=bond_batch)

        edge_vec = data.pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh

    def build_conv_graph(self, data):

        radius_edges = radius_graph(data.pos, self.max_radius, data.batch)
        edge_index = torch.cat([data.edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data.edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_edge_features, device=data.x.device)
        ], 0)

        node_sigma = torch.log(data.node_sigma / self.sigma_min) / np.log(self.sigma_max / self.sigma_min) * 10000
        node_sigma_emb = get_timestep_embedding(node_sigma, self.sigma_embed_dim)

        edge_sigma_emb = node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data.x, node_sigma_emb], 1)

        src, dst = edge_index
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


# Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
