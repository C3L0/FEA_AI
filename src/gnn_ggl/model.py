import torch
import torch.nn as nn

# class MessageLayer(nn.Module):
#     """The 'Brain' of the GNN - calculates node-to-node interactions."""
#
#     def __init__(self, dim):
#         super().__init__()
#         # Input = Node_i + Node_j + Edge_ij (64 + 64 + 2 = 130)
#         self.msg_mlp = nn.Sequential(
#             nn.Linear(dim * 2 + 2, dim),
#             nn.LayerNorm(dim),
#             nn.ReLU(),
#             nn.Linear(dim, dim),
#         )
#         self.up_mlp = nn.Sequential(
#             nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.ReLU()
#         )
#
#     def forward(self, x, edge_index, edge_attr):
#         row, col = edge_index
#         # Gather sender/receiver and edge geometry
#         m = self.msg_mlp(torch.cat([x[row], x[col], edge_attr], dim=1))
#         # Sum messages at receiver
#         aggr = torch.zeros(x.size(0), m.size(1), device=x.device)
#         aggr.index_add_(0, row, m)
#         # Update node state with residual connection
#         return x + self.up_mlp(torch.cat([x, aggr], dim=1))
#
#
# class SolidGNN(nn.Module):
#     """Main GNN architecture."""
#
#     def __init__(self, edge_index, edge_attr, hidden_dim=64, layers=25):
#         super().__init__()
#         self.edge_index = edge_index
#         self.edge_attr = edge_attr
#         self.enc = nn.Linear(3, hidden_dim)
#         self.processor = nn.ModuleList(
#             [MessageLayer(hidden_dim) for _ in range(layers)]
#         )
#         self.dec = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 2))
#
#     def forward(self, x_batch):
#         batch_out = []
#         for b in range(x_batch.shape[0]):
#             h = self.enc(x_batch[b])
#             for layer in self.processor:
#                 h = layer(h, self.edge_index, self.edge_attr)
#             batch_out.append(self.dec(h))
#         return torch.stack(batch_out)


class MessageLayer(nn.Module):
    """The 'Brain' of the GNN - calculates node-to-node interactions."""

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
        # Gather sender/receiver and edge geometry
        # x[row] is the receiver, x[col] is the sender
        m = self.msg_mlp(torch.cat([x[row], x[col], edge_attr], dim=1))

        # Sum messages at receiver
        aggr = torch.zeros(x.size(0), m.size(1), device=x.device)
        aggr.index_add_(0, row, m)

        # Update node state with residual connection
        return x + self.up_mlp(torch.cat([x, aggr], dim=1))


class SolidMechanicsGNN_V2(nn.Module):
    def __init__(self, edge_index, edge_attr, hidden_dim, layers):
        super().__init__()

        # register_buffer ensures these move to the GPU automatically with the model
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)

        # Encoder: Projects raw [Fx, Fy, IsFixed] to hidden dimension
        self.encoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU())

        # Processor: Multiple layers of message passing
        self.layers = nn.ModuleList([MessageLayer(hidden_dim) for _ in range(layers)])

        # Decoder: Projects back to [u, v] displacements
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x_batch):
        batch_size = x_batch.shape[0]
        out_list = []

        for b in range(batch_size):
            h = self.encoder(x_batch[b])

            for layer in self.layers:
                # Use the buffers registered in __init__
                h = layer(h, self.edge_index, self.edge_attr)

            out = self.decoder(h)
            out_list.append(out)

        return torch.stack(out_list)
