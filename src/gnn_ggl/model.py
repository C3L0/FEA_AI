import torch
import torch.nn as nn


class MessageLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Input = Node_i + Node_j + Edge_ij (dim + dim + 2)
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.up_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        m = self.msg_mlp(torch.cat([x[row], x[col], edge_attr], dim=1))

        aggr = torch.zeros(x.size(0), m.size(1), device=x.device)
        aggr.index_add_(0, row, m)

        return x + self.up_mlp(torch.cat([x, aggr], dim=1))


class SolidMechanicsGNN_V3(nn.Module):
    def __init__(self, edge_index, edge_attr, hidden_dim, layers, input_dim=5):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)

        # L'encodeur prend maintenant 5 caractéristiques (Fx, Fy, Fix, E, nu)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        self.layers = nn.ModuleList([MessageLayer(hidden_dim) for _ in range(layers)])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x_batch):
        batch_size = x_batch.shape[0]
        out_list = []
        for b in range(batch_size):
            h = self.encoder(x_batch[b])
            for layer in self.layers:
                h = layer(h, self.edge_index, self.edge_attr)
            out = self.decoder(h)
            out_list.append(out)
        return torch.stack(out_list)
