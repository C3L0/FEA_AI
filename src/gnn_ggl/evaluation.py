import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Import your project classes
from src.gnn_ggl.data import CantileverMeshDataset
from src.gnn_ggl.model import SolidMechanicsGNN_V3


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def visualize_gnn_results(model, dataset, device, sample_idx=0):
    """
    The visualization function you requested.
    """
    model.eval()

    # 1. Get a sample from the dataset
    test_x, test_y_gt = dataset[sample_idx]

    # 2. Predict using the model
    with torch.no_grad():
        # Add batch dimension and move to device
        input_batch = test_x.unsqueeze(0).to(device)
        pred_y = model(input_batch).squeeze(0).cpu().numpy()

    test_y_gt = test_y_gt.numpy()
    coords = dataset.coords
    edges = dataset.edge_index.cpu().numpy()

    # 3. Calculate Error
    error = np.linalg.norm(test_y_gt - pred_y, axis=1)

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    amp = 10.0  # Amplification factor to see the curve

    # --- Wireframe Comparison ---
    gt_pos = coords + test_y_gt * amp
    pr_pos = coords + pred_y * amp

    # Draw wireframes
    for i in range(edges.shape[1]):
        n1, n2 = edges[0, i], edges[1, i]
        ax1.plot(
            [gt_pos[n1, 0], gt_pos[n2, 0]],
            [gt_pos[n1, 1], gt_pos[n2, 1]],
            "b-",
            alpha=0.3,
        )
        ax1.plot(
            [pr_pos[n1, 0], pr_pos[n2, 0]],
            [pr_pos[n1, 1], pr_pos[n2, 1]],
            "r--",
            alpha=0.6,
        )

    ax1.set_title(
        f"Comparison: Blue (Truth) vs Red (GNN Prediction) - Sample {sample_idx}"
    )
    ax1.axis("equal")

    # --- Error Heatmap ---
    scatter = ax2.scatter(pr_pos[:, 0], pr_pos[:, 1], c=error, cmap="viridis", s=30)
    plt.colorbar(scatter, ax=ax2, label="Error Magnitude")
    ax2.set_title("Error Distribution (Heatmap)")
    ax2.axis("equal")

    plt.tight_layout()
    plt.show()


def visualize_solid_mesh(model, dataset, device, sample_idx=0, amp=15.0):
    model.eval()
    test_x, test_y_gt = dataset[sample_idx]

    with torch.no_grad():
        pred_y = model(test_x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

    test_y_gt = test_y_gt.numpy()
    coords = dataset.coords
    edges = dataset.edge_index.cpu().numpy()

    # Calcul de la magnitude du déplacement (pour la couleur)
    disp_magnitude = np.linalg.norm(pred_y, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 1. Dessiner le maillage original (gris très clair)
    for i in range(edges.shape[1]):
        n1, n2 = edges[0, i], edges[1, i]
        ax.plot(
            [coords[n1, 0], coords[n2, 0]],
            [coords[n1, 1], coords[n2, 1]],
            color="gray",
            alpha=0.1,
            lw=0.5,
        )

    # 2. Dessiner le maillage déformé (le SOLIDE)
    new_pos = coords + pred_y * amp

    # On trace les lignes entre les nœuds déformés
    for i in range(edges.shape[1]):
        n1, n2 = edges[0, i], edges[1, i]
        ax.plot(
            [new_pos[n1, 0], new_pos[n2, 0]],
            [new_pos[n1, 1], new_pos[n2, 1]],
            color="black",
            alpha=0.3,
            lw=1.0,
        )

    # 3. Ajouter la colorimétrie (Heatmap de déplacement)
    sc = ax.scatter(
        new_pos[:, 0],
        new_pos[:, 1],
        c=disp_magnitude,
        cmap="jet",
        s=40,
        zorder=3,
        edgecolors="none",
    )

    plt.colorbar(sc, label="Magnitude du déplacement (m)")
    ax.set_title(
        f"Visualisation Solide - Déformation Amplifiée {amp}x\n(Couleur = Intensité du déplacement)"
    )
    ax.set_aspect("equal")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()


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
    model = SolidMechanicsGNN_V3(
        edge_index=dataset.edge_index,
        edge_attr=dataset.edge_attr,
        hidden_dim=cfg["model"]["hidden_dim"],
        layers=cfg["model"]["layers"],
    ).to(device)

    # 4. Load Saved Weights
    model_path = os.path.join(cfg["env"]["save_path"], "gnn_v3.pth")
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
    visualize_solid_mesh(model, dataset, device, sample_idx=2)


if __name__ == "__main__":
    run_evaluation()
