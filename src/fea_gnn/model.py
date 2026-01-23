import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GlobalContextLayer(nn.Module):
    """Permet une communication instantanée entre tous les nœuds (Global pooling)."""

    def __init__(self, dim):
        super().__init__()
        self.node_to_global = nn.Sequential(
            nn.Linear(dim, dim), nn.LeakyReLU(0.2), nn.Linear(dim, dim)
        )
        self.global_to_node = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x, batch):
        # x: [N, dim], batch: [N] (indique à quelle sim appartient chaque nœud)
        # On calcule la moyenne par simulation
        from torch_geometric.nn import global_mean_pool

        global_summary = global_mean_pool(
            self.node_to_global(x), batch
        )  # [BatchSize, dim]

        # On rediffuse l'info globale à chaque nœud
        gate = self.global_to_node(global_summary)  # [BatchSize, dim]
        return x * gate[batch]  # Indexation par batch pour revenir à [N, dim]


class MessageLayer(MessagePassing):
    """Couche de propagation locale utilisant l'API MessagePassing de PyG."""

    def __init__(self, dim):
        # On agrège par somme pour respecter la nature additive des forces/flux
        super().__init__(aggr="add")
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),  # On peut ajouter edge_attr ici si besoin
            nn.LeakyReLU(0.2),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.up_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.LeakyReLU(0.2), nn.LayerNorm(dim)
        )

    def forward(self, x, edge_index):
        # x: [N, dim], edge_index: [2, E]
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i sont les nœuds cibles, x_j les nœuds sources
        return self.msg_mlp(torch.cat([x_i, x_j], dim=-1))

    def update(self, aggr_out, x):
        # aggr_out: [N, dim], x: [N, dim]
        return x + self.up_mlp(torch.cat([x, aggr_out], dim=-1))


class HybridPhysicsGNN(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=4, input_dim=7):
        """
        input_dim=7 : x, y, E, nu, Fx, Fy, isFixed (depuis ton pipeline)
        """
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList(
            [MessageLayer(hidden_dim) for _ in range(n_layers)]
        )
        self.global_layers = nn.ModuleList(
            [GlobalContextLayer(hidden_dim) for _ in range(n_layers)]
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),  # Sortie : ux, uy
        )

    def forward(self, data):
        # data est un objet Batch de PyG
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = self.encoder(x)

        for gnn, glob in zip(self.gnn_layers, self.global_layers):
            # 1. Propagation locale (Physique de proche en proche)
            h = gnn(h, edge_index)
            # 2. Contexte global (Équilibre global de la structure)
            h = glob(h, batch)

        return self.decoder(h)


# import torch
# import torch.nn as nn
#
#
# class GlobalContextLayer(nn.Module):
#     """Permet une communication instantanée entre tous les nœuds."""
#
#     def __init__(self, dim):
#         super().__init__()
#         self.node_to_global = nn.Sequential(
#             nn.Linear(dim, dim), nn.LeakyReLU(0.2), nn.Linear(dim, dim)
#         )
#         self.global_to_node = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
#
#     def forward(self, x):
#         # x est [N, dim]
#         global_summary = torch.mean(self.node_to_global(x), dim=0, keepdim=True)
#         gate = self.global_to_node(global_summary)
#         return x * gate
#
#
# class MessageLayer(nn.Module):
#     """Couche de propagation locale (Physique de proche en proche)."""
#
#     def __init__(self, dim):
#         super().__init__()
#         self.msg_mlp = nn.Sequential(
#             nn.Linear(dim * 2 + 2, dim),
#             nn.LeakyReLU(0.2),
#             nn.LayerNorm(dim),
#             nn.Linear(dim, dim),
#         )
#         self.up_mlp = nn.Sequential(
#             nn.Linear(dim * 2, dim), nn.LeakyReLU(0.2), nn.LayerNorm(dim)
#         )
#
#     def forward(self, x, edge_index, edge_attr):
#         row, col = edge_index
#         # Ici x est 2D [N, dim], donc x[row] est 2D [Edges, dim]
#         # edge_attr est 2D [Edges, 2]. Tout concorde !
#         m = self.msg_mlp(torch.cat([x[row], x[col], edge_attr], dim=1))
#
#         aggr = torch.zeros(x.size(0), m.size(1), device=x.device)
#         aggr.index_add_(0, row, m)
#
#         return x + self.up_mlp(torch.cat([x, aggr], dim=1))
#
#
# class HybridPhysicsGNN(nn.Module):
#     def __init__(self, edge_index, edge_attr, hidden_dim, layers, input_dim=5):
#         super().__init__()
#         self.register_buffer("edge_index", edge_index)
#         self.register_buffer("edge_attr", edge_attr)
#
#         self.encoder = nn.Linear(input_dim, hidden_dim)
#
#         self.gnn_layers = nn.ModuleList()
#         self.global_layers = nn.ModuleList()
#
#         for _ in range(layers):
#             self.gnn_layers.append(MessageLayer(hidden_dim))
#             self.global_layers.append(GlobalContextLayer(hidden_dim))
#
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, 2)
#         )
#
#     def forward(self, x_batch):
#         batch_size = x_batch.shape[0]
#         out_list = []
#
#         # Correction de la dimension : on traite chaque échantillon du batch
#         for b in range(batch_size):
#             h = self.encoder(x_batch[b])  # h devient [N, hidden_dim]
#
#             for i in range(len(self.gnn_layers)):
#                 # 1. Local
#                 h = self.gnn_layers[i](h, self.edge_index, self.edge_attr)
#                 # 2. Global
#                 h = self.global_layers[i](h)
#
#             out = self.decoder(h)
#             out_list.append(out)
#
#         return torch.stack(out_list)
