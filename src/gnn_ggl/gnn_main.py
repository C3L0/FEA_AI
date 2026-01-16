import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.gnn_ggl.data import CantileverMeshDataset
from src.gnn_ggl.evaluation import visualize_gnn_results
from src.gnn_ggl.model import SolidMechanicsGNN_V2


def load_config(config_path="config.yaml"):
    """
    Loads a YAML configuration file.
    """
    if not os.path.exists(config_path):
        # Default configuration structure
        default_config = {
            "geometry": {"nx": 25, "ny": 5, "length": 2.0, "height": 0.5},
            "model": {"hidden_dim": 64, "layers": 12},
            "training": {
                "epochs": 200,
                "batch_size": 32,
                "learning_rate": 0.002,
                "data_weight": 1.0,
                "physics_weight": 0.01,
            },
            "env": {"device": "cuda", "save_path": "./models"},
        }
        with open(config_path, "w") as file:
            yaml.dump(default_config, file)
        return default_config

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def run_simulation():
    cfg = load_config()

    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # 3. Initialize Dataset using geometry settings from config
    dataset = CantileverMeshDataset(
        num_samples=800,
    )

    # 4. Initialize Model using architecture settings from config
    model = SolidMechanicsGNN_V2(
        edge_index=dataset.edge_index,
        edge_attr=dataset.edge_attr,
        hidden_dim=cfg["model"]["hidden_dim"],
        layers=cfg["model"]["layers"],
    ).to(device)

    # 5. Training Setup using training settings from config
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"], shuffle=True
    )

    print(f"Model initialized with {cfg['model']['layers']} layers.")

    # Training Loop
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0

        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            # Forward pass
            out = model(bx)

            # Loss Calculation (Uses weights defined in config)
            loss_mse = F.mse_loss(out, by)

            # Here you could integrate the physics_loss using:
            # loss_phys = calculate_potential_energy(...)
            # total_loss = cfg['training']['data_weight'] * loss_mse + cfg['training']['physics_weight'] * loss_phys

            total_loss = cfg["training"]["data_weight"] * loss_mse
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Average Loss: {epoch_loss / len(loader):.6f}")

    # Save the model
    save_dir = cfg["env"]["save_path"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.state_dict(), os.path.join(save_dir, "gnn_v2.pth"))
    print(f"Training complete. Model saved to {save_dir}/gnn_v2.pth")


if __name__ == "__main__":
    run_simulation()
    # visualize_gnn_results()
