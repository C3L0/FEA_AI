import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class GlobalContextLayer(nn.Module):
    """Allows an instant communication between each nodes"""

    def __init__(self, dim):
        super().__init__()
        self.node_to_global = nn.Sequential(
            nn.Linear(dim, dim), nn.LeakyReLU(0.2), nn.Linear(dim, dim)
        )
        self.global_to_node = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, x, batch):
        latent = self.node_to_global(x)
        g_mean = global_mean_pool(latent, batch)
        g_max = global_max_pool(latent, batch)
        global_summary = torch.cat([g_mean, g_max], dim=1)
        gate = self.global_to_node(global_summary)
        return x * gate[batch]


class MessageLayer(nn.Module):
    """
    Explicitly manage edges attributs
    """

    def __init__(self, dim, heads=4):
        super().__init__()

        # edge_dim=3 [dx, dy, length] concat=False we average the heads to keep the stable dimension
        self.gat = GATv2Conv(
            in_channels=dim,
            out_channels=dim,
            heads=heads,
            concat=False,
            edge_dim=3,
            dropout=0.0,  # no dropout in pure physic regression
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.LeakyReLU(0.2), nn.Linear(dim * 2, dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # attention
        h = x + self.gat(x, edge_index, edge_attr=edge_attr)
        h = self.norm1(h)

        # feedforward
        h = h + self.ffn(h)
        h = self.norm2(h)

        return h


class HybridPhysicsGNN(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=4, input_dim=7):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList(
            [MessageLayer(hidden_dim, heads=4) for _ in range(n_layers)]
        )
        self.global_layers = nn.ModuleList(
            [GlobalContextLayer(hidden_dim) for _ in range(n_layers)]
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        h = self.encoder(x)

        for gnn, glob in zip(self.gnn_layers, self.global_layers):
            h = gnn(h, edge_index, edge_attr)
            h = glob(h, batch)

        return self.decoder(h)
