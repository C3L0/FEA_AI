# import torch
# import matplotlib.pyplot as plt
# import numpy as np
#
# def visualize_gnn_results(model, dataset, device, sample_idx=0):
#     model.eval()
#
#     # 1. Get Data
#     # test_x: [Num_Nodes, 3] (Input features)
#     # test_y_gt: [Num_Nodes, 2] (True displacements from analytical solver)
#     test_x, test_y_gt = dataset[sample_idx]
#
#     # 2. Predict
#     with torch.no_grad():
#         # Add batch dim, send to device, then bring back to CPU and numpy
#         pred_y = model(test_x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
#
#     test_y_gt = test_y_gt.numpy()
#     coords = dataset.coords
#     edges = dataset.edge_index.cpu().numpy()
#
#     # 3. Calculate Euclidean Error per node
#     # Error = sqrt((u_true - u_pred)^2 + (v_true - v_pred)^2)
#     error = np.linalg.norm(test_y_gt - pred_y, axis=1)
#
#     # 4. Setup Plot
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
#     amp = 10.0 # Amplification factor
#
#     # --- TOP PLOT: WIREFRAME COMPARISON ---
#     # Draw original (undeformed) mesh in light gray
#     for i in range(edges.shape[1]):
#         n1, n2 = edges[0, i], edges[1, i]
#         ax1.plot([coords[n1,0], coords[n2,0]], [coords[n1,1], coords[n2,1]], 'gray', alpha=0.1, lw=0.5)
#
#     # Draw Ground Truth (Blue) and GNN Prediction (Red)
#     gt_pos = coords + test_y_gt * amp
#     pr_pos = coords + pred_y * amp
#
#     for i in range(edges.shape[1]):
#         n1, n2 = edges[0, i], edges[1, i]
#         ax1.plot([gt_pos[n1,0], gt_pos[n2,0]], [gt_pos[n1,1], gt_pos[n2,1]], 'b-', alpha=0.3, label='GT' if i==0 else "")
#         ax1.plot([pr_pos[n1,0], pr_pos[n2,0]], [pr_pos[n1,1], pr_pos[n2,1]], 'r--', alpha=0.6, label='GNN' if i==0 else "")
#
#     ax1.set_title(f"Wireframe Comparison (Amplified {amp}x)")
#     ax1.legend()
#     ax1.axis('equal')
#
#     # --- BOTTOM PLOT: ERROR HEATMAP ---
#     # This shows WHERE the model is failing
#     scatter = ax2.scatter(pr_pos[:, 0], pr_pos[:, 1], c=error, cmap='viridis', s=30)
#     plt.colorbar(scatter, ax=ax2, label="Absolute Error (m)")
#
#     # Draw wireframe for context
#     for i in range(edges.shape[1]):
#         n1, n2 = edges[0, i], edges[1, i]
#         ax2.plot([pr_pos[n1,0], pr_pos[n2,0]], [pr_pos[n1,1], pr_pos[n2,1]], 'black', alpha=0.1, lw=0.5)
#
#     ax2.set_title("Prediction Error Heatmap (Where is the AI struggling?)")
#     ax2.axis('equal')
#
#     plt.tight_layout()
#     plt.show()

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Import your project classes
from src.gnn_ggl.data import CantileverMeshDataset
from src.gnn_ggl.model import SolidMechanicsGNN_V2


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


def run_evaluation():
    # 1. Load Configuration
    cfg = load_config()
    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")

    # 2. Re-create Dataset (needs to be same geometry as training)
    dataset = CantileverMeshDataset(
        num_samples=10,  # We only need a few for evaluation
        nx=cfg["geometry"]["nx"],
        ny=cfg["geometry"]["ny"],
        length=cfg["geometry"]["length"],
        height=cfg["geometry"]["height"],
    )

    # 3. Initialize Model Architecture
    model = SolidMechanicsGNN_V2(
        edge_index=dataset.edge_index,
        edge_attr=dataset.edge_attr,
        hidden_dim=cfg["model"]["hidden_dim"],
        layers=cfg["model"]["layers"],
    ).to(device)

    # 4. Load Saved Weights
    model_path = os.path.join(cfg["env"]["save_path"], "gnn_v2.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load the state dictionary into the model
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # 5. Visualize!
    # Change sample_idx to see different force scenarios
    visualize_gnn_results(model, dataset, device, sample_idx=2)


if __name__ == "__main__":
    run_evaluation()
