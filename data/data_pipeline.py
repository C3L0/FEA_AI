import os

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import coalesce
from tqdm import tqdm


class PlateHoleDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        nodes_csv="db.csv",
        topo_csv="connectivity.csv",
        transform=None,
        pre_transform=None,
    ):
        """
        root: Dossier où les données transformées seront stockées.
        """
        self.nodes_csv = nodes_csv
        self.topo_csv = topo_csv
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # Les fichiers sources nécessaires
        return [self.nodes_csv, self.topo_csv]

    @property
    def processed_file_names(self):
        # Le nom du fichier final généré
        return ["dataset.pt"]

    def download(self):
        # Pas de téléchargement nécessaire ici, les fichiers sont locaux
        pass

    def process(self):
        """
        Cette méthode n'est exécutée qu'une seule fois.
        Elle lit les CSV et crée le fichier .pt
        """
        print(f"Chargement des fichiers depuis {self.root}...")
        df_nodes = pd.read_csv(os.path.join(self.root, self.nodes_csv))
        df_topo = pd.read_csv(os.path.join(self.root, self.topo_csv))

        data_list = []
        sim_ids = df_nodes["SimulationID"].unique()

        for sim_id in tqdm(sim_ids, desc="Conversion des simulations en graphes"):
            # 1. Extraire les données de cette simulation
            nodes_sim = df_nodes[df_nodes["SimulationID"] == sim_id].sort_values(
                "NodeID"
            )
            topo_sim = df_topo[df_topo["SimulationID"] == sim_id]

            # 2. Features des nœuds (X)
            # Normalisation : E est souvent très grand (GPa), on le divise par 100e9
            node_features = nodes_sim[
                ["x", "y", "E", "nu", "Fx", "Fy", "isFixed"]
            ].values.copy()
            node_features[:, 2] /= 1e9  # E en GPa pour la stabilité

            x = torch.tensor(node_features, dtype=torch.float)

            # 3. Cibles (Y) : Déplacements (ux, uy)
            # Souvent multiplié par 1000 (m -> mm) pour aider l'IA à voir des chiffres > 0.001
            node_labels = nodes_sim[["ux", "uy"]].values * 1000.0
            y = torch.tensor(node_labels, dtype=torch.float)

            # 4. Construction des arêtes (Edge Index)
            # On transforme les triangles (n1, n2, n3) en paires d'arêtes
            edges = []
            for _, row in topo_sim.iterrows():
                n1, n2, n3 = int(row["n1"]), int(row["n2"]), int(row["n3"])
                edges.extend(
                    [[n1, n2], [n2, n1], [n2, n3], [n3, n2], [n3, n1], [n1, n3]]
                )

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = coalesce(edge_index)  # Nettoyage des doublons

            # 5. Création de l'objet Data
            data = Data(x=x, edge_index=edge_index, y=y)
            data.sim_id = int(sim_id)

            data_list.append(data)

        # Sauvegarde optimisée pour InMemoryDataset
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Dataset traité sauvegardé dans {self.processed_paths[0]}")


if __name__ == "__main__":
    # Utilisation :
    # On suppose que db.csv et connectivity.csv sont dans le dossier actuel '.'
    # Le dataset sera créé dans un dossier 'data_processed'
    try:
        dataset = PlateHoleDataset(root=".")

        print("\n--- Infos Dataset ---")
        print(f"Nombre de simulations : {len(dataset)}")
        print(f"Nombre de features : {dataset.num_features}")

        # Exemple d'accès
        first_graph = dataset[0]
        print(
            f"Premier graphe : {first_graph.num_nodes} nœuds, {first_graph.num_edges} arêtes"
        )

    except Exception as e:
        print(f"Erreur lors de la création du dataset : {e}")
