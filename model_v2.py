import random
import numpy as np
import os
from src import config as cfg
from src.eagle import EagleDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import JumpingKnowledge, TopKPooling
from torch_geometric import data
from torch_scatter import scatter_sum
from datetime import datetime


class PositionalEncoding(nn.Module):
    def __init__(self: object,
                 d_model: int,
                 max_len: int=5000) -> None:
        """
        Define the PositionalEncoding class to add positional information to input data
        """
        super(PositionalEncoding, self).__init__()

        # Create a positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # Apply sine and cosine functions to the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self: object,
                x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to the input tensor
        x = x + self.pe[:, :x.size(1), :x.size(2)]
        return x


class Normalizer(nn.Module):
    def __init__(self: object,
                 input_size: int) -> None:
        """
        Define the Normalizer class to normalize input features
        """
        super(Normalizer, self).__init__()

        # So no grad would be saved into memory
        self.register_buffer('accumulation', torch.zeros(input_size))
        self.register_buffer('accumulation_squared', torch.zeros(input_size))
        self.register_buffer('mean', torch.zeros(input_size))
        self.register_buffer('std', torch.ones(input_size))
        self.max_accumulation = 1e7
        self.count = 0

    def forward(self: object,
                x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        feature_dim = original_shape[-1]
        x = x.view(-1, feature_dim)
        if self.training:
            if self.count < self.max_accumulation:
                self.count += x.size(0)
                self.accumulation[:feature_dim] += torch.sum(x, dim=0)
                self.accumulation_squared[:feature_dim] += torch.sum(x ** 2, dim=0)
                self.mean[:feature_dim] = self.accumulation[:feature_dim] / (self.count + 1e-8)
                self.std[:feature_dim] = torch.sqrt(self.accumulation_squared[:feature_dim] / (self.count + 1e-8) - self.mean[:feature_dim] ** 2)

        mean = self.mean[:feature_dim]
        std = self.std[:feature_dim]
        x = (x - mean) / (std + 1e-8)
        return x.view(*original_shape)

    def inverse(self: object,
                x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        feature_dim = original_shape[-1]
        x = x.view(-1, feature_dim)
        mean = self.mean[:feature_dim].unsqueeze(0)
        std = self.std[:feature_dim].unsqueeze(0)
        x = x * (std + 1e-8) + mean
        return x.view(*original_shape)


class MLP(nn.Module):
    def __init__(self: object,
                 input_size: int,
                 output_size: int,
                 layer_norm: bool=True) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.layer_norm = nn.LayerNorm(output_size) if layer_norm else nn.Identity()

    def forward(self: object,
                x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.layer_norm(x)


class Encoder(nn.Module):
    def __init__(self: object,
                 state_size: int) -> None:
        """
        Define an Encoder class for processing input data
        """
        super(Encoder, self).__init__()
        self.normalize_edges = Normalizer(3)
        self.normalize_nodes = Normalizer(9 + state_size)
        self.fv = MLP(input_size=9 + state_size, output_size=128)
        self.fe = MLP(input_size=3, output_size=128)

    def forward(self: object,
                mesh_pos: torch.Tensor,
                edges: torch.Tensor,
                node_type: torch.Tensor,
                velocity: torch.Tensor) -> torch.Tensor:
        V = torch.cat([velocity, node_type], dim=-1)
        V = self.fv(self.normalize_nodes(V))

        # Calculate edge features based on mesh positions
        senders = torch.gather(mesh_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, 2))
        receivers = torch.gather(mesh_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, 2))
        distance = senders - receivers
        norm = torch.sqrt((distance ** 2).sum(-1, keepdim=True))
        E = torch.cat([distance, norm], dim=-1)
        E = self.fe(self.normalize_edges(E))
        return V, E


class GAT(nn.Module):
    def __init__(self: object,
                 node_size: int,
                 output_size: int):
        super(GAT, self).__init__()

        # Define edge transformation and attention mechanisms
        self.f_edge = nn.Linear(in_features=node_size, out_features=output_size, bias=False)
        self.attention = nn.Sequential(nn.Linear(in_features=output_size * 2 + 128, out_features=1),
                                       nn.LeakyReLU(0.2))

    def forward(self: object,
                V: torch.Tensor,
                E: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, V.shape[-1]))

        h_sender = self.f_edge(senders)
        h_receiver = self.f_edge(receivers)

        attention = self.attention(torch.cat([h_sender, h_receiver, E], dim=-1))
        attention = torch.exp(attention - torch.max(attention, dim=1, keepdim=True)[0])
        col = edges[..., 0].unsqueeze(-1)

        numerator = scatter_sum(attention * h_sender, col.repeat(1, 1, h_sender.shape[-1]), dim=-2)
        denominator = scatter_sum(attention, col.repeat(1, 1, attention.shape[-1]), dim=-2)
        h = numerator / (denominator + 1e-8)

        return h


class multiHeadGAT(nn.Module):
    def __init__(self: object,
                 node_size: int,
                 output_size: int,
                 n_heads: int=4) -> None:
        super(multiHeadGAT, self).__init__()
        self.gat = nn.ModuleList([GAT(node_size, output_size // n_heads) for _ in range(n_heads)])

    def forward(self, V, E, edges):
        heads = [gat(V, E, edges) for gat in self.gat]
        return torch.cat(heads, dim=-1)


class Processor(nn.Module):
    def __init__(self: object,
                 N: int=15,
                 n_heads: int=4) -> None:
        """
        Define a Processor class that applies multiple GAT layers
        """
        super(Processor, self).__init__()
        self.gat = nn.ModuleList([multiHeadGAT(node_size=128, output_size=128, n_heads=n_heads) for _ in range(N)])

    def forward(self: object,
                V: torch.Tensor,
                E: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        for gat in self.gat:
            v = gat(V, E, edges)
            V = V + v
        return V


class GNN(nn.Module):
    def __init__(self: object,
                 N: int=8,
                 state_size: int=4,
                 n_heads: int=4,
                 max_len: int=5000) -> None:
        super(GNN, self).__init__()

        self.pos_encoder = PositionalEncoding(3, max_len)
        self.encoder = Encoder(state_size)
        self.processor = Processor(N, n_heads=n_heads)
        self.decoder = MLP(input_size=128, output_size=4, layer_norm=False)
        self.normalizer_output = Normalizer(state_size)
        self.pool = TopKPooling(128, ratio=0.8)
        self.jk = JumpingKnowledge(mode='cat', channels=128, num_layers=N)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.dropout = nn.Dropout(0.8)

    def forward(self: object,
                mesh_pos: torch.Tensor,
                edges: torch.Tensor,
                state: torch.Tensor,
                node_type: torch.Tensor) -> torch.Tensor:
        state_hat, output_hat = [state[:, 0]], []
        target = state[:, 1:] - state[:, :-1]
        target = self.normalizer_output(target)

        for t in range(1, state.shape[1]):
            mesh_pos_t = self.pos_encoder(mesh_pos[:, t - 1])
            V, E = self.encoder(mesh_pos_t, edges[:, t - 1], node_type[:, t - 1], state_hat[-1])
            V = self.processor(V, E, edges[:, t - 1])

            # Had to squeeze and unsqueeze to make the code work, failed because of the batch dimension
            # V = V.squeeze(0)
            # pool_edges = edges[:, t - 1].squeeze(0)
            # V, edge_index, _, batch, perm, _ = self.pool(V, pool_edges)
            # V = V.unsqueeze(0)
            # edge_index = edge_index.unsqueeze(0)
            V = self.jk([V])
            V, _ = self.lstm(V)
            V = self.dropout(V)
            next_output = self.decoder(V)
            output_denormalized = self.normalizer_output.inverse(next_output)
            next_state = state_hat[-1] + output_denormalized

            mask = (node_type[:, t, :, 4] == 1) | (node_type[:, t, :, 6] == 1) | (
                        node_type[:, t, :, 2] == 1)
            next_state[mask, :] = state[:, t][mask, :]
            state_hat.append(next_state)
            output_hat.append(next_output)
            break

        state_hat = torch.stack(state_hat, dim=1)
        output_hat = torch.stack(output_hat, dim=1)

        return state_hat, output_hat, target


def test_GNN(model, dataloader):
    """
    Used only for testing
    """
    model.eval()
    time_start = datetime.now()
    for i, data in enumerate(dataloader):
        mesh_pos = data["mesh_pos"].to(cfg.DEVICE)
        edges = data['edges'].to(cfg.DEVICE).long()
        velocity = data["velocity"].to(cfg.DEVICE)
        pressure = data["pressure"].to(cfg.DEVICE)
        node_type = data["node_type"].to(cfg.DEVICE)
        print(f"mesh_pos: {mesh_pos.shape},\nedges: {edges.shape},\nvelocity: {velocity.shape},\npressure: {pressure.shape},\nnode_type: {node_type.shape}")

        state = torch.cat([velocity, pressure], dim=-1)
        print(f"state: {state.shape}")

        mask = torch.ones_like(mesh_pos)[..., 0]
        print(f"mask: {mask.shape}")

        state_hat, output_hat, target = model(mesh_pos, edges, state, node_type)
        print(f"state_hat: {state_hat.shape}, output_hat: {output_hat.shape}, target: {target.shape}")
        print(f'Real state: {state},\nPredicted state: {state_hat}')

    time_end = datetime.now()
    print(f"Time taken: {time_end - time_start}")


if __name__ == '__main__':
    dataset = EagleDataset(data_path=cfg.DATA_DIR,
                           mode='train',
                           window_length=100,
                           apply_onehot=True,
                           with_cells=False,
                           with_cluster=True,
                           n_cluster = 40,
                           normalize=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    model = GNN().to(cfg.DEVICE)
    test_GNN(model, dataloader)