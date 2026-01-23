import os

import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader  # Import spécifique à PyG

# Imports internes
from src.fea_gnn.data_loader import get_dataset  # CantileverMeshDataset
from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config


def calculate_pinn_loss(pred_u, data, weight_physics):
    """
    Calcule la pénalité physique (PINN) de manière vectorisée.
    Plus besoin de boucle 'for b in range(batch_size)'.
    """
    if weight_physics <= 0:
        return torch.tensor(0.0, device=pred_u.device)

    # Dans PyG, edge_index contient déjà les connexions pour tout le batch
    row, col = data.edge_index

    # 1. Énergie de déformation (Stiffness locale)
    # On récupère E (index 3 dans tes features : Fx, Fy, isFixed, E, nu)
    # Note : Ajuste l'index si ton input_dim a changé (ex: index 2 si x,y sont absents)
    E = data.x[row, 3]

    # Différence de déplacement entre voisins sur tout le batch
    diff_u = pred_u[row] - pred_u[col]

    # Énergie interne : Moyenne de E * ||delta_u||²
    # On utilise .view(-1) pour s'assurer que E multiplie correctement chaque arête
    strain_energy = torch.mean(E.view(-1) * torch.norm(diff_u, dim=1) ** 2)

    # 2. Travail des forces (Fy est à l'index 1)
    # Travail = Force * Déplacement
    fy = data.x[:, 1]
    uy = pred_u[:, 1]
    external_work = torch.mean(fy * uy)

    # La physique cherche à minimiser (Interne - Externe)
    return (strain_energy - external_work) * weight_physics


def train_model():
    # 1. Configuration
    cfg = load_config()
    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Démarrage de l'entraînement Hybride sur : {device}")

    # 2. Données (Utilisation du DataLoader de PyTorch Geometric)
    # dataset = CantileverMeshDataset(
    #     num_samples=1000,
    #     nx=cfg["geometry"]["nx"],
    #     ny=cfg["geometry"]["ny"],
    #     length=cfg["geometry"]["length"],
    #     height=cfg["geometry"]["height"],
    #     E_range=cfg["material"]["youngs_modulus_range"],
    #     nu_range=cfg["material"]["poissons_ratio_range"],
    # )
    dataset = get_dataset(root=".")

    # CRUCIAL : On utilise le loader de torch_geometric pour gérer le batching des graphes
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    # 3. Modèle Hybride
    # On s'assure que les buffers (edge_index, edge_attr) sont bien initialisés
    model = HybridPhysicsGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=15, factor=0.5
    )

    # 4. Boucle d'entraînement
    epochs = cfg["training"]["epochs"]
    p_weight = cfg["training"].get("physics_weight", 0.05)

    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Prédiction : Le modèle prend l'objet 'data' complet
            out = model(data)

            # --- CALCUL DES PERTES ---
            # Perte 1 : Erreur de données (comparaison avec data.y)
            loss_data = F.mse_loss(out, data.y)

            # Perte 2 : Erreur Physique (PINN)
            loss_phys = calculate_pinn_loss(out, data, p_weight)

            # Somme des deux
            total_loss = loss_data + loss_phys

            total_loss.backward()

            # Gradient Clipping pour éviter les explosions numériques (fréquent en PINN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_epoch_loss += total_loss.item()

        avg_loss = total_epoch_loss / len(loader)
        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            print(
                f"Époque {epoch:03d} | Loss Totale: {avg_loss:.6f} "
                f"(Data: {loss_data.item():.6f}, Phys: {loss_phys.item():.6f})"
            )

    # 5. Sauvegarde
    os.makedirs(cfg["env"]["save_path"], exist_ok=True)
    save_file = os.path.join(cfg["env"]["save_path"], "gnn_hybrid.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Entraînement terminé. Modèle sauvegardé sous : {save_file}")


if __name__ == "__main__":
    train_model()

# import os
# import time
#
# import torch
# import torch.nn.functional as F
# import yaml
# from torch.utils.data import DataLoader
#
# # Imports internes
# from src.fea_gnn.data_loader import CantileverMeshDataset
# from src.fea_gnn.model import HybridPhysicsGNN
# from src.fea_gnn.utils import load_config
#
#
# def calculate_pinn_loss(pred_u, edge_index, edge_attr, features, weight_physics):
#     """
#     Calcule la pénalité physique (PINN).
#     Force la continuité et le respect du module d'élasticité E.
#     """
#     batch_size = pred_u.shape[0]
#     total_phys_loss = 0
#     row, col = edge_index
#
#     for b in range(batch_size):
#         u = pred_u[b]  # Déplacements prédits [N, 2]
#         feat = features[b]  # Caractéristiques [N, 5]
#
#         # 1. Énergie de déformation : On veut que les voisins bougent de façon cohérente
#         # Le module de Young E est à l'index 3
#         E = feat[row, 3].unsqueeze(1)
#         diff_u = u[row] - u[col]
#
#         # On pénalise les variations brutales (le carré de la différence)
#         # C'est ce qui va "lisser" ta poutre et enlever l'aspect haché
#         strain_energy = torch.mean(E * torch.norm(diff_u, dim=1) ** 2)
#
#         # 2. Travail des forces : Le déplacement doit être dans le sens de la force (Fy à l'index 1)
#         external_work = torch.mean(feat[:, 1] * u[:, 1])
#
#         # La physique cherche à minimiser (Énergie interne - Travail externe)
#         total_phys_loss += strain_energy - external_work
#
#     return (total_phys_loss / batch_size) * weight_physics
#
#
# def train_model():
#     # 1. Configuration
#     cfg = load_config()
#     device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")
#     print(f"Démarrage de l'entraînement Hybride sur : {device}")
#
#     # 2. Données
#     dataset = CantileverMeshDataset(
#         num_samples=1000,
#         nx=cfg["geometry"]["nx"],
#         ny=cfg["geometry"]["ny"],
#         length=cfg["geometry"]["length"],
#         height=cfg["geometry"]["height"],
#         E_range=cfg["material"]["youngs_modulus_range"],
#         nu_range=cfg["material"]["poissons_ratio_range"],
#     )
#     loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
#
#     # 3. Modèle Hybride (Local + Global)
#     model = HybridPhysicsGNN(
#         edge_index=dataset.edge_index,
#         edge_attr=dataset.edge_attr,
#         hidden_dim=cfg["model"]["hidden_dim"],
#         layers=cfg["model"]["layers"],
#         input_dim=cfg["model"]["input_dim"],
#     ).to(device)
#
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=cfg["training"]["learning_rate"]
#     )
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", patience=15, factor=0.5
#     )
#
#     # 4. Boucle d'entraînement
#     epochs = cfg["training"]["epochs"]
#     p_weight = cfg["training"].get(
#         "physics_weight", 0.05
#     )  # On récupère le poids depuis la config
#
#     for epoch in range(epochs):
#         model.train()
#         total_epoch_loss = 0
#
#         for batch_x, batch_y in loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             optimizer.zero_grad()
#
#             # Prédiction
#             out = model(batch_x)
#
#             # --- CALCUL DES PERTES ---
#             # Perte 1 : Erreur par rapport aux données (MSE)
#             loss_data = F.mse_loss(out, batch_y)
#
#             # Perte 2 : Erreur Physique (PINN)
#             loss_phys = calculate_pinn_loss(
#                 out, model.edge_index, model.edge_attr, batch_x, p_weight
#             )
#
#             # Somme des deux
#             total_loss = loss_data + loss_phys
#
#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#
#             total_epoch_loss += total_loss.item()
#
#         avg_loss = total_epoch_loss / len(loader)
#         scheduler.step(avg_loss)
#
#         if epoch % 10 == 0:
#             print(
#                 f"Époque {epoch:03d} | Loss Totale: {avg_loss:.6f} (Data: {loss_data.item():.6f}, Phys: {loss_phys.item():.6f})"
#             )
#
#     # 5. Sauvegarde
#     os.makedirs(cfg["env"]["save_path"], exist_ok=True)
#     torch.save(
#         model.state_dict(), os.path.join(cfg["env"]["save_path"], "gnn_hybrid.pth")
#     )
#     print("Entraînement terminé et modèle sauvegardé.")
#
#
# if __name__ == "__main__":
#     train_model()
