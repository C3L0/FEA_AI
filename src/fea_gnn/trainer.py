import csv
import os

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# Imports internes
from src.fea_gnn.data_loader import get_dataset
from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config


def calculate_pinn_loss(pred_u, data, weight_physics):
    """
    PINN Loss stabilisée pour éviter l'enroulement (curling).
    Formulation par équilibre des forces par nœud.
    """
    if weight_physics <= 1e-7:
        return torch.tensor(0.0, device=pred_u.device)

    row, col = data.edge_index
    E = data.x[row, 2].view(-1, 1)
    nu = data.x[row, 3].view(-1, 1)

    # 1. Géométrie des arêtes (dist_x, dist_y, length)
    if data.edge_attr is not None and data.edge_attr.shape[1] >= 3:
        edge_len = data.edge_attr[:, 2].view(-1, 1) + 1e-6
    else:
        edge_len = torch.ones_like(E)

    # 2. Calcul des déformations relatives (Strain)
    diff_u = pred_u[row] - pred_u[col]
    strain = diff_u / edge_len

    # 3. Force Interne (Loi de Hooke simplifiée)
    # On utilise une raideur k = E / L
    k = E / edge_len
    f_internal_edges = -k * diff_u

    # --- AJOUT DE L'EFFET DE POISSON STABILISÉ ---
    # On réduit l'effet de Poisson PINN au minimum (0.05) pour éviter les torsions
    # L'IA apprendra l'essentiel du Poisson via les données FEniCS
    strain_x = strain[:, 0].view(-1, 1)
    strain_y = strain[:, 1].view(-1, 1)
    f_poisson = torch.cat([-nu * strain_y, -nu * strain_x], dim=1) * k * 0.05

    total_edge_force = f_internal_edges + f_poisson

    # 4. Agrégation (Newton : Somme des forces sur chaque nœud)
    f_total_nodes = torch.zeros_like(pred_u)
    f_total_nodes.index_add_(0, row, total_edge_force)

    # 5. Équilibre avec Force Externe
    # On utilise le facteur d'équilibrage ~15.0
    f_ext = data.x[:, 4:6] * 15.0

    # Résidu d'équilibre : F_int + F_ext = 0
    # On utilise Huber pour ignorer les nœuds aberrants (comme ceux qui s'enroulent)
    res_equilibre = F.huber_loss(f_total_nodes, f_ext, delta=1.0)

    # 6. Conditions aux limites (Fixations murales)
    is_fixed = data.x[:, 6].view(-1, 1)
    boundary_loss = torch.mean((is_fixed * pred_u) ** 2)

    return (res_equilibre + 20.0 * boundary_loss) * weight_physics


def train_model():
    cfg = load_config()
    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Démarrage de l'entraînement de stabilisation sur : {device}")

    dataset = get_dataset(root="data/")
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    model = HybridPhysicsGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )
    # Scheduler plus réactif
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    log_path = os.path.join(cfg["env"]["save_path"], "training_history.csv")
    os.makedirs(cfg["env"]["save_path"], exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_loss", "data_loss", "phys_loss", "lr"])

    epochs = cfg["training"]["epochs"]
    target_phys_weight = cfg["training"].get(
        "physics_weight", 0.01
    )  # On baisse le poids physique

    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0
        total_data_loss = 0
        total_phys_loss = 0

        # Curriculum : On laisse les données diriger pendant 50 époques
        if epoch < 50:
            current_p_weight = 0.0
        else:
            current_p_weight = target_phys_weight

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)

            # Perte de données L1 (Forte sensibilité)
            loss_data = F.l1_loss(out, data.y)
            # Perte physique (Stabilisatrice)
            loss_phys = calculate_pinn_loss(out, data, current_p_weight)

            total_loss = loss_data + loss_phys
            total_loss.backward()

            # Clipping de gradient TRÈS strict (0.1) pour empêcher le maillage de se plier
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            total_epoch_loss += total_loss.item()
            total_data_loss += loss_data.item()
            total_phys_loss += loss_phys.item()

        avg_loss = total_epoch_loss / len(loader)
        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            print(
                f"Époque {epoch:03d} | Total: {avg_loss:.6f} | Data: {total_data_loss / len(loader):.6f} | Phys: {total_phys_loss / len(loader):.6f}"
            )

    save_file = os.path.join(cfg["env"]["save_path"], "gnn_hybrid.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Entraînement terminé. Modèle sauvegardé sous : {save_file}")


if __name__ == "__main__":
    train_model()

# import csv
# import os
#
# import torch
# import torch.nn.functional as F
# from torch_geometric.loader import DataLoader
#
# # Imports internes
# from src.fea_gnn.data_loader import get_dataset
# from src.fea_gnn.model import HybridPhysicsGNN
# from src.fea_gnn.utils import load_config
#
#
# def calculate_pinn_loss(pred_u, data, weight_physics):
#     """
#     PINN Loss stabilisée pour éviter l'enroulement (curling).
#     Formulation par équilibre des forces par nœud.
#     """
#     if weight_physics <= 1e-7:
#         return torch.tensor(0.0, device=pred_u.device)
#
#     row, col = data.edge_index
#     E = data.x[row, 2].view(-1, 1)
#     nu = data.x[row, 3].view(-1, 1)
#
#     # 1. Géométrie des arêtes (dist_x, dist_y, length)
#     if data.edge_attr is not None and data.edge_attr.shape[1] >= 3:
#         edge_len = data.edge_attr[:, 2].view(-1, 1) + 1e-6
#     else:
#         edge_len = torch.ones_like(E)
#
#     # 2. Calcul des déformations relatives (Strain)
#     diff_u = pred_u[row] - pred_u[col]
#     strain = diff_u / edge_len
#
#     # 3. Force Interne (Loi de Hooke simplifiée)
#     # On utilise une raideur k = E / L
#     k = E / edge_len
#     f_internal_edges = -k * diff_u
#
#     # --- AJOUT DE L'EFFET DE POISSON STABILISÉ ---
#     # On réduit l'effet de Poisson PINN au minimum (0.05) pour éviter les torsions
#     # L'IA apprendra l'essentiel du Poisson via les données FEniCS
#     strain_x = strain[:, 0].view(-1, 1)
#     strain_y = strain[:, 1].view(-1, 1)
#     f_poisson = torch.cat([-nu * strain_y, -nu * strain_x], dim=1) * k * 0.05
#
#     total_edge_force = f_internal_edges + f_poisson
#
#     # 4. Agrégation (Newton : Somme des forces sur chaque nœud)
#     f_total_nodes = torch.zeros_like(pred_u)
#     f_total_nodes.index_add_(0, row, total_edge_force)
#
#     # 5. Équilibre avec Force Externe
#     # On utilise le facteur d'équilibrage ~15.0
#     f_ext = data.x[:, 4:6] * 15.0
#
#     # Résidu d'équilibre : F_int + F_ext = 0
#     # On utilise Huber pour ignorer les nœuds aberrants (comme ceux qui s'enroulent)
#     res_equilibre = F.huber_loss(f_total_nodes, f_ext, delta=1.0)
#
#     # 6. Conditions aux limites (Fixations murales)
#     is_fixed = data.x[:, 6].view(-1, 1)
#     boundary_loss = torch.mean((is_fixed * pred_u) ** 2)
#
#     return (res_equilibre + 20.0 * boundary_loss) * weight_physics
#
#
# def train_model():
#     cfg = load_config()
#     device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")
#     os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
#
#     print(f"Démarrage de l'entraînement de stabilisation sur : {device}")
#
#     dataset = get_dataset(root="data/")
#     loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
#
#     model = HybridPhysicsGNN(
#         hidden_dim=cfg["model"]["hidden_dim"],
#         n_layers=cfg["model"]["layers"],
#         input_dim=cfg["model"]["input_dim"],
#     ).to(device)
#
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=cfg["training"]["learning_rate"]
#     )
#     # Scheduler plus réactif
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", patience=10, factor=0.5
#     )
#
#     log_path = os.path.join(cfg["env"]["save_path"], "training_history.csv")
#     os.makedirs(cfg["env"]["save_path"], exist_ok=True)
#     with open(log_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["epoch", "total_loss", "data_loss", "phys_loss", "lr"])
#
#     epochs = cfg["training"]["epochs"]
#     target_phys_weight = cfg["training"].get(
#         "physics_weight", 0.01
#     )  # On baisse le poids physique
#
#     for epoch in range(epochs):
#         model.train()
#         total_epoch_loss = 0
#         total_data_loss = 0
#         total_phys_loss = 0
#
#         # Curriculum : On laisse les données diriger pendant 50 époques
#         if epoch < 50:
#             current_p_weight = 0.0
#         else:
#             current_p_weight = target_phys_weight
#
#         for data in loader:
#             data = data.to(device)
#             optimizer.zero_grad()
#             out = model(data)
#
#             # Perte de données L1 (Forte sensibilité)
#             loss_data = F.l1_loss(out, data.y)
#             # Perte physique (Stabilisatrice)
#             loss_phys = calculate_pinn_loss(out, data, current_p_weight)
#
#             total_loss = loss_data + loss_phys
#             total_loss.backward()
#
#             # Clipping de gradient TRÈS strict (0.1) pour empêcher le maillage de se plier
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
#             optimizer.step()
#
#             total_epoch_loss += total_loss.item()
#             total_data_loss += loss_data.item()
#             total_phys_loss += loss_phys.item()
#
#         avg_loss = total_epoch_loss / len(loader)
#         scheduler.step(avg_loss)
#
#         if epoch % 10 == 0:
#             print(
#                 f"Époque {epoch:03d} | Total: {avg_loss:.6f} | Data: {total_data_loss / len(loader):.6f} | Phys: {total_phys_loss / len(loader):.6f}"
#             )
#
#     save_file = os.path.join(cfg["env"]["save_path"], "gnn_hybrid.pth")
#     torch.save(model.state_dict(), save_file)
#     print(f"Entraînement terminé. Modèle sauvegardé sous : {save_file}")
#
#
# if __name__ == "__main__":
#     train_model()
