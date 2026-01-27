import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import coalesce
from tqdm import tqdm

from src.fea_gnn.utils import load_config


class PlateHoleDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        nodes_csv="db.csv",
        topo_csv="connectivity.csv",
        transform=None,
        pre_transform=None,
    ):
        self.nodes_csv = nodes_csv
        self.topo_csv = topo_csv
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [self.nodes_csv, self.topo_csv]

    @property
    def processed_file_names(self):
        return ["dataset.pt"]

    def process(self):
        print(f"Chargement depuis {self.root}...")
        df_nodes = pd.read_csv(os.path.join(self.root, self.nodes_csv))
        df_topo = pd.read_csv(os.path.join(self.root, self.topo_csv))

        # --- CONFIGURATION DU SCALING ---
        # 1.000.000 par défaut (micromètres) pour aider l'IA
        TARGET_SCALE = 1_000_000.0
        print(f"!!! SCALING ACTIF : x{TARGET_SCALE} !!!")

        cfg = load_config()
        norm_cfg = cfg.get("normalization", {})

        data_list = []
        sim_ids = df_nodes["SimulationID"].unique()

        for sim_id in tqdm(sim_ids, desc="Conversion"):
            nodes_sim = df_nodes[df_nodes["SimulationID"] == sim_id].sort_values(
                "NodeID"
            )
            topo_sim = df_topo[df_topo["SimulationID"] == sim_id]

            # Features
            feat = nodes_sim[["x", "y", "E", "nu", "Fx", "Fy", "isFixed"]].to_numpy(
                dtype=float
            )
            pos_physical = torch.tensor(feat[:, 0:2], dtype=torch.float)

            # Normalisation Inputs (Standard)
            feat[:, 0] /= float(norm_cfg.get("x", 1.0))
            feat[:, 1] /= float(norm_cfg.get("y", 1.0))
            feat[:, 2] /= float(norm_cfg.get("E", 1.0))
            feat[:, 4] /= float(norm_cfg.get("force", 1.0))  # Fx
            feat[:, 5] /= float(norm_cfg.get("force", 1.0))  # Fy

            x = torch.tensor(feat, dtype=torch.float)

            # Cibles avec SCALING FORCE
            y_values = nodes_sim[["ux", "uy"]].to_numpy(dtype=float) * TARGET_SCALE
            y = torch.tensor(y_values, dtype=torch.float)

            if y.shape[1] != 2:
                continue

            # Topology (Vectorisée)
            cells = topo_sim[["n1", "n2", "n3"]].to_numpy(dtype=int)
            idx_pairs = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 0], [0, 2]]
            edges_list = [cells[:, p] for p in idx_pairs]
            all_edges = np.vstack(edges_list)

            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            edge_index = coalesce(edge_index)

            # Edge Attributes
            row, col = edge_index
            edge_vector = pos_physical[col] - pos_physical[row]
            edge_len = torch.norm(edge_vector, dim=1, keepdim=True)
            edge_attr = torch.cat([edge_vector, edge_len], dim=1)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.sim_id = int(sim_id)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Dataset généré avec succès.")


if __name__ == "__main__":
    PlateHoleDataset(root="data/")
