import os

#
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

#
# Import your project classes
from src.gnn_ggl.data import CantileverMeshDataset
from src.gnn_ggl.model import HybridPhysicsGNN
from src.utils import load_config
from src.visualizer import (visualize_comparison_fea, visualize_gnn_results,
                            visualize_solid_mesh)

# compute mse error compute energy residual evaluate on test set


def run_evaluation():
    # 1. Load Configuration
    cfg = load_config()
    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")

    # 2. Re-create Dataset (needs to be same geometry as training)
    dataset = CantileverMeshDataset(
        num_samples=800,
        nx=cfg["geometry"]["nx"],
        ny=cfg["geometry"]["ny"],
        length=cfg["geometry"]["length"],
        height=cfg["geometry"]["height"],
        E_range=cfg["material"]["youngs_modulus_range"],
        nu_range=cfg["material"]["poissons_ratio_range"],
    )

    # 3. Initialize Model Architecture
    model = HybridPhysicsGNN(
        edge_index=dataset.edge_index,
        edge_attr=dataset.edge_attr,
        hidden_dim=cfg["model"]["hidden_dim"],
        layers=cfg["model"]["layers"],
    ).to(device)

    # 4. Load Saved Weights
    model_path = os.path.join(cfg["env"]["save_path"], "gnn_hybrid.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load the state dictionary into the model
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # 5. Visualize!
    # Change sample_idx to see different force scenarios
    # visualize_gnn_results(model, dataset, device, sample_idx=2)
    # visualize_solid_mesh(model, dataset, device, sample_idx=2)
    visualize_comparison_fea(model, dataset, device, sample_idx=2)


if __name__ == "__main__":
    run_evaluation()
