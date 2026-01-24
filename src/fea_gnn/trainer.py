import csv
import os

import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

# Imports internes
from src.fea_gnn.data_loader import get_dataset
from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config


def calculate_pinn_loss(pred_u, data, weight_physics):
    """
    Calcule le résidu d'équilibre des forces (F_interne - F_externe)^2.
    Toutes les valeurs ici sont normalisées.
    """
    if weight_physics <= 0:
        return torch.tensor(0.0, device=pred_u.device)

    row, col = data.edge_index

    # E_norm est à l'index 2 (déjà /210e9 dans le pipeline)
    E = data.x[row, 2].view(-1, 1)

    # Loi de Hooke simplifiée : Force = K * delta_u
    # pred_u est en mm, donc diff_u est en mm.
    diff_u = pred_u[row] - pred_u[col]

    # On simule la raideur locale
    internal_forces_edges = E * diff_u

    # Somme des forces arrivant sur chaque nœud
    f_int = torch.zeros_like(pred_u)
    f_int.index_add_(0, row, internal_forces_edges)

    # Force externe (déjà /1e7 dans le pipeline)
    f_ext = data.x[:, 4:6]

    # Équilibre : f_int - f_ext = 0 (au carré pour rester positif)
    residu_equilibre = torch.mean((f_int - f_ext) ** 2)

    # Contrainte Dirichlet : Points fixes ne bougent pas
    is_fixed = data.x[:, 6].view(-1, 1)
    boundary_loss = torch.mean((is_fixed * pred_u) ** 2)

    return (residu_equilibre + 10.0 * boundary_loss) * weight_physics


def train_model():
    cfg = load_config()
    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Démarrage de l'entraînement Hybride sur : {device}")

    # Utilisation de root='.' pour trouver le dossier data/processed
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=15, factor=0.5
    )

    log_path = os.path.join(cfg["env"]["save_path"], "training_history.csv")
    os.makedirs(cfg["env"]["save_path"], exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_loss", "data_loss", "phys_loss", "lr"])

    epochs = cfg["training"]["epochs"]
    # On utilise un petit poids physique au début pour stabiliser
    p_weight = cfg["training"].get("physics_weight", 0.01)

    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0
        total_data_loss = 0
        total_phys_loss = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)

            loss_data = F.mse_loss(out, data.y)

            # Calcul de la perte physique corrigée
            loss_phys = calculate_pinn_loss(out, data, p_weight)

            total_loss = loss_data + loss_phys
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_epoch_loss += total_loss.item()
            total_data_loss += loss_data.item()
            total_phys_loss += loss_phys.item()

        avg_loss = total_epoch_loss / len(loader)
        scheduler.step(avg_loss)

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    avg_loss,
                    total_data_loss / len(loader),
                    total_phys_loss / len(loader),
                    optimizer.param_groups[0]["lr"],
                ]
            )

        if epoch % 10 == 0:
            print(
                f"Époque {epoch:03d} | Loss: {avg_loss:.6f} (Data: {total_data_loss / len(loader):.6f}, Phys: {total_phys_loss / len(loader):.6f})"
            )

    save_file = os.path.join(cfg["env"]["save_path"], "gnn_hybrid.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Entraînement terminé. Modèle sauvegardé : {save_file}")


if __name__ == "__main__":
    train_model()
