import os

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
        print(f"Chargement des fichiers depuis {self.root}...")
        df_nodes = pd.read_csv(os.path.join(self.root, self.nodes_csv))
        df_topo = pd.read_csv(os.path.join(self.root, self.topo_csv))

        cfg = load_config()
        norm_cfg = cfg["normalization"]

        data_list = []
        sim_ids = df_nodes["SimulationID"].unique()

        for sim_id in tqdm(sim_ids, desc="Conversion des simulations"):
            nodes_sim = df_nodes[df_nodes["SimulationID"] == sim_id].sort_values(
                "NodeID"
            )
            topo_sim = df_topo[df_topo["SimulationID"] == sim_id]

            # 1. Features
            cols = ["x", "y", "E", "nu", "Fx", "Fy", "isFixed"]
            feat = nodes_sim[cols].to_numpy(dtype=float)

            # On garde une copie des coordonnées NON normalisées pour calculer les distances physiques
            pos_physical = torch.tensor(feat[:, 0:2], dtype=torch.float)

            # Normalisation
            feat[:, 0] /= float(norm_cfg["x"])
            feat[:, 1] /= float(norm_cfg["y"])
            feat[:, 2] /= float(norm_cfg["E"])
            feat[:, 3] /= float(norm_cfg["nu"])
            feat[:, 4] /= float(norm_cfg["force"])
            feat[:, 5] /= float(norm_cfg["force"])

            x = torch.tensor(feat, dtype=torch.float)

            # 2. Cibles
            target_scale = float(norm_cfg.get("target_scale", 1000.0))
            y_values = nodes_sim[["ux", "uy"]].to_numpy(dtype=float) * target_scale
            y = torch.tensor(y_values, dtype=torch.float)

            # 3. Arêtes & Distances
            edges = []
            cells = topo_sim[["n1", "n2", "n3"]].to_numpy(dtype=int)
            for cell in cells:
                n1, n2, n3 = cell
                edges.extend(
                    [[n1, n2], [n2, n1], [n2, n3], [n3, n2], [n3, n1], [n1, n3]]
                )

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            # On nettoie les doublons
            edge_index = coalesce(edge_index)

            # --- CALCUL CRITIQUE DES ATTRIBUTS D'ARÊTE ---
            # On calcule le vecteur distance réel (en mètres) entre les nœuds connectés
            row, col = edge_index
            # pos_physical est en mètres
            edge_vector = pos_physical[col] - pos_physical[row]  # [dx, dy]

            # On ajoute aussi la longueur (norme) comme feature d'arête
            edge_len = torch.norm(edge_vector, dim=1, keepdim=True)
            # edge_attr : [dx, dy, length]
            edge_attr = torch.cat([edge_vector, edge_len], dim=1)

            # On crée l'objet Data avec edge_attr
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.sim_id = int(sim_id)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Dataset sauvegardé : {self.processed_paths[0]}")


if __name__ == "__main__":
    PlateHoleDataset(root="data/")
