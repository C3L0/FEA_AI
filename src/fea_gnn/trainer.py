import csv
import os

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.fea_gnn.data_loader import get_dataset
from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config

TARGET_SCALE = 1_000_000.0


# Added to prevent the AI from 'cheating'
def weighted_mse_loss(pred, target, weight_factor=50.0):
    """
    weighted average that penalizes mobile errors
    """
    # MSE
    loss = (pred - target) ** 2

    # default weight -> 1.0
    weights = torch.ones_like(target)

    # Identification of the area taht move with a scale factor of 1e6
    mask = target.abs() > 0.5

    # Increasing the weight of critical area
    weights[mask] *= weight_factor

    return (loss * weights).mean()


### every is scaled otherwise the value were too little
def calculate_pinn_loss(pred_u, data, weight_physics, target_scale):
    """
    PINN Loss calculation
    """
    if weight_physics <= 1e-9:
        return torch.tensor(0.0, device=pred_u.device)

    row, col = data.edge_index
    E = data.x[row, 2].view(-1, 1)
    nu = data.x[row, 3].view(-1, 1)

    # Edge geometry
    if data.edge_attr is not None and data.edge_attr.shape[1] >= 3:
        edge_len = data.edge_attr[:, 2].view(-1, 1) + 1e-6
    else:
        edge_len = torch.ones_like(E)

    # Strain : deformation calculated on scaled prediction
    diff_u = pred_u[row] - pred_u[col]
    strain = diff_u / edge_len

    # Internal force (K * u)
    # k : local stiffness u : scaled F_int : scaled
    k = E / edge_len
    f_internal_edges = -k * diff_u

    # Poisson effect (little)
    strain_x = strain[:, 0].view(-1, 1)
    strain_y = strain[:, 1].view(-1, 1)
    f_poisson = torch.cat([-nu * strain_y, -nu * strain_x], dim=1) * k * 0.05
    total_edge_force = f_internal_edges + f_poisson

    # Sum of force at the nodes
    f_total_nodes = torch.zeros_like(pred_u)
    f_total_nodes.index_add_(0, row, total_edge_force)

    # Balance with external force (scaled)
    f_ext = data.x[:, 4:6] * target_scale
    # empiric balance factor (needed for the convergence)
    f_ext = f_ext * 15.0

    # Residu : F_int + F_ext = 0
    res_equilibre = F.l1_loss(f_total_nodes, f_ext)

    # conditions at the limits
    is_fixed = data.x[:, 6].view(-1, 1)
    boundary_loss = torch.mean((is_fixed * pred_u) ** 2)

    return (res_equilibre + 100.0 * boundary_loss) * weight_physics


def train_model():
    cfg = load_config()
    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    print(
        f"Starting the training (Scale={cfg['normalization']['target_scale']}) on : {device}"
    )

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
        optimizer, mode="min", patience=25, factor=0.5
    )

    log_path = os.path.join(cfg["env"]["save_path"], "training_history.csv")
    os.makedirs(cfg["env"]["save_path"], exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_loss", "data_loss", "phys_loss", "lr"])

    epochs = cfg["training"]["epochs"]
    target_phys_weight = cfg["training"]["physics_weight"]

    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0
        total_data_loss = 0
        total_phys_loss = 0

        # Curriculum learning for physic
        if epoch < 20:
            current_p_weight = 0.0
        else:
            # Progessive increase in physics
            progress = min(1.0, (epoch - 20) / 50)
            current_p_weight = target_phys_weight * progress

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)

            loss_data = weighted_mse_loss(out, data.y, weight_factor=50.0)

            loss_phys = calculate_pinn_loss(
                out, data, current_p_weight, cfg["normalization"]["target_scale"]
            )

            total_loss = loss_data + loss_phys
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_epoch_loss += total_loss.item()
            total_data_loss += loss_data.item()
            total_phys_loss += loss_phys.item()

        avg_loss = total_epoch_loss / len(loader)

        scheduler.step(avg_loss)

        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Ep {epoch:03d} | Tot: {avg_loss:.4f} | Data: {total_data_loss / len(loader):.4f} | Phys: {total_phys_loss / len(loader):.4f} | LR: {current_lr:.1e}"
            )

            # Writing in a csv
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

    ### rename the model name but all together
    save_file = os.path.join(cfg["env"]["save_path"], "gnn_hybrid_scaled.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved : {save_file}")


if __name__ == "__main__":
    train_model()
