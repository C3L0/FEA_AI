import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    nx, ny = dataset.nx, dataset.ny
    
    # Coordonnées déformées
    new_pos = coords + pred_y * amp
    
    # Calcul de la magnitude du déplacement pour la couleur
    magnitude = np.linalg.norm(pred_y, axis=1)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # --- DESSIN DES QUADS (Éléments finis) ---
    # Au lieu de tracer des lignes individuelles, on trace des carrés remplis
    for i in range(ny - 1):
        for j in range(nx - 1):
            # Index des 4 nœuds formant un carré
            idx1 = i * nx + j
            idx2 = i * nx + (j + 1)
            idx3 = (i + 1) * nx + (j + 1)
            idx4 = (i + 1) * nx + j
            
            # Points déformés correspondants
            quad_coords = [new_pos[idx1], new_pos[idx2], new_pos[idx3], new_pos[idx4]]
            
            # Couleur moyenne du quad basée sur le déplacement
            avg_mag = np.mean([magnitude[idx1], magnitude[idx2], magnitude[idx3], magnitude[idx4]])
            
            # On dessine le polygone
            polygon = patches.Polygon(quad_coords, closed=True, 
                                      linewidth=0.5, edgecolor='black', 
                                      facecolor=plt.cm.jet(avg_mag / (np.max(magnitude) + 1e-9)),
                                      alpha=0.8)
            ax.add_patch(polygon)

    # Paramètres esthétiques
    ax.set_xlim(-0.1, dataset.length + 0.5)
    ax.set_ylim(-dataset.height - 0.5, dataset.height + 0.2)
    ax.set_aspect('equal')
    
    # Barre de couleur personnalisée
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=np.max(magnitude)))
    plt.colorbar(sm, label="Déplacement (m)", ax=ax, shrink=0.6)

    # Affichage des propriétés du matériau pour ce sample
    # (On récupère E et nu depuis les features du nœud 0)
    E_norm = test_x[0, 3].item()
    nu = test_x[0, 4].item()
    ax.set_title(f"Simulation GNN - Rendu Solide (FEA Style)\n"
                 f"Matériau : E_norm={E_norm:.2f}, nu={nu:.2f} | Amplification: {amp}x", 
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
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
