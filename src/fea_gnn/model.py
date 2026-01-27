import torch
import torch.nn as nn
from torch_geometric.nn import (MessagePassing, global_max_pool,
                                global_mean_pool)


class GlobalContextLayer(nn.Module):
    """Permet une communication instantanée entre tous les nœuds (Global pooling)."""

    def __init__(self, dim):
        super().__init__()
        # Encodeur des features locales avant pooling
        self.node_to_global = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        # Projection du contexte global vers les nœuds
        # MODIFICATION : On retire la Sigmoid qui écrasait le signal entre 0 et 1.
        # On utilise une projection linéaire pour permettre des ajustements positifs ou négatifs.
        self.global_to_node = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),  # Sortie linéaire pour l'addition résiduelle
        )

    def forward(self, x, batch):
        # x: [N, dim], batch: [N] (indique à quelle sim appartient chaque nœud)

        # 1. Extraction des caractéristiques latentes
        latent = self.node_to_global(x)

        # 2. Calcul du résumé global (Moyenne + Max pour capturer les pics de force)
        g_mean = global_mean_pool(latent, batch)
        g_max = global_max_pool(latent, batch)

        # [BatchSize, dim * 2]
        global_summary = torch.cat([g_mean, g_max], dim=1)

        # 3. On rediffuse l'info globale à chaque nœud
        # context : [BatchSize, dim]
        context = self.global_to_node(global_summary)

        # MODIFICATION MAJEURE : Addition (Skip Connection) au lieu de multiplication.
        # Cela permet au contexte d'informer le nœud sans l'uniformiser.
        return x + context[batch]


class MessageLayer(MessagePassing):
    """Couche de propagation locale utilisant l'API MessagePassing de PyG."""

    def __init__(self, dim):
        # On agrège par somme pour respecter la nature additive des forces
        super().__init__(aggr="add")

        # Remplacement de LeakyReLU par GELU pour plus de fluidité
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

        self.up_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim)
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
        Architecture hybride combinant propagation locale et contexte global.
        input_dim=7 : [x, y, E, nu, Fx, Fy, isFixed]
        """
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList(
            [MessageLayer(hidden_dim) for _ in range(n_layers)]
        )
        self.global_layers = nn.ModuleList(
            [GlobalContextLayer(hidden_dim) for _ in range(n_layers)]
        )

        # Décodeur final
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),  # Activation plus douce
            nn.Linear(hidden_dim, 2),  # Sortie brute (ux, uy) - PAS D'ACTIVATION ICI
        )

    def forward(self, data):
        # data est un objet Batch de PyG
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = self.encoder(x)

        for gnn, glob in zip(self.gnn_layers, self.global_layers):
            # 1. Propagation locale (Stiffness / Équilibre local)
            h = gnn(h, edge_index)
            # 2. Contexte global (Rediffusion des forces aux extrémités)
            h = glob(h, batch)

        return self.decoder(h)
