import csv
import os

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.fea_gnn.data_loader import get_dataset
from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config

# --- CONSTANTE DE SCALING ---
# Doit correspondre à celle utilisée dans data_pipeline.py
TARGET_SCALE = 1_000_000.0


def weighted_mse_loss(pred, target, weight_factor=50.0):
    """
    MSE pondérée qui punit sévèrement les erreurs sur les zones mobiles.
    """
    # Erreur quadratique de base
    loss = (pred - target) ** 2

    # Poids par défaut de 1.0 partout
    weights = torch.ones_like(target)

    # On identifie les zones qui bougent "pour de vrai"
    # Avec un scale de 1e6, 1.0 = 1 micromètre. C'est un bon seuil de bruit.
    mask = target.abs() > 0.5

    # On augmente le poids sur ces zones critiques
    weights[mask] *= weight_factor

    return (loss * weights).mean()


def calculate_pinn_loss(pred_u, data, weight_physics):
    """
    PINN Loss adaptée au scaling x1e6.
    On travaille directement dans l'espace scalé pour la stabilité du gradient.
    """
    if weight_physics <= 1e-9:
        return torch.tensor(0.0, device=pred_u.device)

    row, col = data.edge_index
    E = data.x[row, 2].view(-1, 1)
    nu = data.x[row, 3].view(-1, 1)

    # 1. Géométrie des arêtes
    if data.edge_attr is not None and data.edge_attr.shape[1] >= 3:
        edge_len = data.edge_attr[:, 2].view(-1, 1) + 1e-6
    else:
        edge_len = torch.ones_like(E)

    # 2. Déformations (Strain) calculées sur les prédictions scalées
    diff_u = pred_u[row] - pred_u[col]
    strain = diff_u / edge_len

    # 3. Force Interne (K * u)
    # k est la raideur locale. Comme u est scalé, F_int sera scalé aussi.
    k = E / edge_len
    f_internal_edges = -k * diff_u

    # Effet Poisson stabilisé (minime)
    strain_x = strain[:, 0].view(-1, 1)
    strain_y = strain[:, 1].view(-1, 1)
    f_poisson = torch.cat([-nu * strain_y, -nu * strain_x], dim=1) * k * 0.05
    total_edge_force = f_internal_edges + f_poisson

    # 4. Somme des forces aux nœuds
    f_total_nodes = torch.zeros_like(pred_u)
    f_total_nodes.index_add_(0, row, total_edge_force)

    # 5. Équilibre avec Force Externe
    # IMPORTANT : Comme F_int est calculé sur u scalé par 1e6,
    # F_int est 1e6 fois trop grand. Il faut aussi scaler F_ext par 1e6.
    f_ext = data.x[:, 4:6] * TARGET_SCALE

    # Facteur d'équilibrage empirique (conserve le * 15.0 si nécessaire pour la convergence)
    f_ext = f_ext * 15.0

    # Résidu : F_int + F_ext = 0
    # On utilise L1 ou Huber pour la robustesse
    res_equilibre = F.l1_loss(f_total_nodes, f_ext)

    # 6. Conditions aux limites (Fixations murales)
    is_fixed = data.x[:, 6].view(-1, 1)
    # On force les nœuds fixes à rester à 0
    boundary_loss = torch.mean((is_fixed * pred_u) ** 2)

    return (res_equilibre + 100.0 * boundary_loss) * weight_physics


def train_model():
    cfg = load_config()
    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Démarrage de l'entraînement (Scale={TARGET_SCALE}) sur : {device}")

    dataset = get_dataset(root="data/")
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    model = HybridPhysicsGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    # Learning Rate un peu plus élevé au début car les valeurs sont grandes
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=25, factor=0.5
    )

    log_path = os.path.join(cfg["env"]["save_path"], "training_history.csv")
    os.makedirs(cfg["env"]["save_path"], exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_loss", "data_loss", "phys_loss", "lr"])

    epochs = cfg["training"]["epochs"]
    target_phys_weight = cfg["training"].get("physics_weight", 0.01)
    # target_phys_weight = 0.0

    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0
        total_data_loss = 0
        total_phys_loss = 0

        # Curriculum learning pour la physique
        if epoch < 20:
            current_p_weight = 0.0
        else:
            # Montée progressive de la physique
            progress = min(1.0, (epoch - 20) / 50)
            current_p_weight = target_phys_weight * progress

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)

            # --- NOUVELLE LOSS PONDÉRÉE ---
            # On utilise Weighted MSE au lieu de L1 pour punir les pics d'erreur
            loss_data = weighted_mse_loss(out, data.y, weight_factor=50.0)

            # Loss Physique (Mise à l'échelle)
            loss_phys = calculate_pinn_loss(out, data, current_p_weight)

            total_loss = loss_data + loss_phys
            total_loss.backward()

            # Gradient Clipping (Important pour la stabilité avec les gros chiffres)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_epoch_loss += total_loss.item()
            total_data_loss += loss_data.item()
            total_phys_loss += loss_phys.item()

        avg_loss = total_epoch_loss / len(loader)

        # Le scheduler se base sur la Loss Totale
        scheduler.step(avg_loss)

        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Ep {epoch:03d} | Tot: {avg_loss:.4f} | Data: {total_data_loss / len(loader):.4f} | Phys: {total_phys_loss / len(loader):.4f} | LR: {current_lr:.1e}"
            )

            # Écriture dans le CSV
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        avg_loss,
                        total_data_loss / len(loader),
                        total_phys_loss / len(loader),
                        current_lr,
                    ]
                )

    save_file = os.path.join(cfg["env"]["save_path"], "gnn_hybrid_scaled.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Modèle sauvegardé : {save_file}")


if __name__ == "__main__":
    train_model()
