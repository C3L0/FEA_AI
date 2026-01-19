import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Internal imports
from src.gnn_ggl.data import CantileverMeshDataset
from src.gnn_ggl.model import SolidMechanicsGNN_V3


def load_config(config_path="config.yaml"):
    """Loads the project configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train_model():
    # 1. Setup Environment & Config
    cfg = load_config()
    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Starting V3 Training on device: {device}")

    # 2. Initialize Material-Aware Dataset
    # We pass the material ranges defined in the config
    dataset = CantileverMeshDataset(
        num_samples=cfg.get("training", {}).get("num_samples", 1000),
        nx=cfg["geometry"]["nx"],
        ny=cfg["geometry"]["ny"],
        length=cfg["geometry"]["length"],
        height=cfg["geometry"]["height"],
        E_range=cfg["material"]["youngs_modulus_range"],
        nu_range=cfg["material"]["poissons_ratio_range"],
    )

    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    # 3. Initialize V3 Model (5 Input Features)
    model = SolidMechanicsGNN_V3(
        edge_index=dataset.edge_index,
        edge_attr=dataset.edge_attr,
        hidden_dim=cfg["model"]["hidden_dim"],
        layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    # 4. Optimizer & Stability Tools
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )

    # Scheduler: Reduces learning rate if loss stops improving
    # Removed 'verbose=True' to fix compatibility with newer PyTorch versions
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=15, factor=0.5
    )

    print(f"Training model with {cfg['model']['layers']} layers...")
    start_time = time.time()

    # 5. Training Loop
    epochs = cfg["training"]["epochs"]
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch_x)

            # Loss Calculation (MSE)
            # Note: batch_y is already scaled (x1000) in the dataset generator
            loss = F.mse_loss(predictions, batch_y)

            loss.backward()

            # STABILITY: Gradient Clipping
            # Prevents "Exploding Gradients" in deep GNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        # Step the scheduler based on average epoch loss
        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

    # 6. Save Model
    save_path = cfg["env"]["save_path"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_name = f"gnn_ly{cfg['model']['layers']}_ep{cfg['training']['epochs']}"
    torch.save(model.state_dict(), os.path.join(save_path, file_name))

    total_time = time.time() - start_time
    print(f"Training Complete! Total time: {total_time:.2f}s")
    print(f"Model saved to {save_path}/{file_name}")


if __name__ == "__main__":
    train_model()
