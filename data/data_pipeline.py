import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from tqdm import tqdm


def create_gnn_dataset(nodes_path="db.csv", connectivity_path="connectivity.csv"):
    """
    Transforme les CSV FEniCS en une liste d'objets Data de PyTorch Geometric.
    """
    print("Chargement des fichiers CSV...")
    df_nodes = pd.read_csv(nodes_path)
    df_topo = pd.read_csv(connectivity_path)

    dataset = []
    sim_ids = df_nodes["SimulationID"].unique()

    print(f"Conversion de {len(sim_ids)} simulations en graphes...")
    for sim_id in tqdm(sim_ids):
        # 1. Extraire les données de cette simulation
        nodes_sim = df_nodes[df_nodes["SimulationID"] == sim_id].sort_values("NodeID")
        topo_sim = df_topo[df_topo["SimulationID"] == sim_id]

        # 2. Features des nœuds (X)
        # On prend : x, y, E, nu, Fx, Fy, isFixed
        node_features = nodes_sim[["x", "y", "E", "nu", "Fx", "Fy", "isFixed"]].values
        x = torch.tensor(node_features, dtype=torch.float)

        # 3. Cibles (Y) : Ce que l'IA doit prédire (déplacements ux, uy)
        node_labels = nodes_sim[["ux", "uy"]].values
        y = torch.tensor(node_labels, dtype=torch.float)

        # 4. Construction des arêtes (Edge Index)
        # On transforme les triangles (n1, n2, n3) en arêtes (n1-n2, n2-n3, n3-n1)
        edges = []
        for _, row in topo_sim.iterrows():
            n1, n2, n3 = int(row["n1"]), int(row["n2"]), int(row["n3"])
            # On ajoute les arêtes dans les deux sens (graphe non orienté)
            edges.extend([[n1, n2], [n2, n1], [n2, n3], [n3, n2], [n3, n1], [n1, n3]])

        # Supprimer les doublons d'arêtes et convertir en tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # On garde uniquement les arêtes uniques

        edge_index = coalesce(edge_index)

        # 5. Créer l'objet Data
        data = Data(x=x, edge_index=edge_index, y=y)
        data.sim_id = sim_id  # Pour garder la trace

        dataset.append(data)

    return dataset


if __name__ == "__main__":
    # Test rapide
    try:
        my_data_list = create_gnn_dataset()
        print(f"\nPipeline réussi !")
        print(f"Nombre de graphes : {len(my_data_list)}")
        print(f"Exemple du premier graphe :")
        print(f" - Nombre de nœuds : {my_data_list[0].num_nodes}")
        print(f" - Nombre d'arêtes : {my_data_list[0].num_edges}")
        print(f" - Features par nœud : {my_data_list[0].num_node_features}")

        # Sauvegarde du dataset traité pour ne pas avoir à relancer les CSV
        torch.save(my_data_list, "processed_dataset.pt")
        print("\nDataset sauvegardé sous 'processed_dataset.pt'")

    except Exception as e:
        print(f"Erreur dans le pipeline : {e}")
