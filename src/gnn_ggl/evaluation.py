import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_gnn_results(model, dataset, device, sample_idx=0):
    model.eval()
    
    # 1. Get Data
    # test_x: [Num_Nodes, 3] (Input features)
    # test_y_gt: [Num_Nodes, 2] (True displacements from analytical solver)
    test_x, test_y_gt = dataset[sample_idx]
    
    # 2. Predict
    with torch.no_grad():
        # Add batch dim, send to device, then bring back to CPU and numpy
        pred_y = model(test_x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    
    test_y_gt = test_y_gt.numpy()
    coords = dataset.coords
    edges = dataset.edge_index.cpu().numpy()

    # 3. Calculate Euclidean Error per node
    # Error = sqrt((u_true - u_pred)^2 + (v_true - v_pred)^2)
    error = np.linalg.norm(test_y_gt - pred_y, axis=1)

    # 4. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    amp = 10.0 # Amplification factor

    # --- TOP PLOT: WIREFRAME COMPARISON ---
    # Draw original (undeformed) mesh in light gray
    for i in range(edges.shape[1]):
        n1, n2 = edges[0, i], edges[1, i]
        ax1.plot([coords[n1,0], coords[n2,0]], [coords[n1,1], coords[n2,1]], 'gray', alpha=0.1, lw=0.5)

    # Draw Ground Truth (Blue) and GNN Prediction (Red)
    gt_pos = coords + test_y_gt * amp
    pr_pos = coords + pred_y * amp

    for i in range(edges.shape[1]):
        n1, n2 = edges[0, i], edges[1, i]
        ax1.plot([gt_pos[n1,0], gt_pos[n2,0]], [gt_pos[n1,1], gt_pos[n2,1]], 'b-', alpha=0.3, label='GT' if i==0 else "")
        ax1.plot([pr_pos[n1,0], pr_pos[n2,0]], [pr_pos[n1,1], pr_pos[n2,1]], 'r--', alpha=0.6, label='GNN' if i==0 else "")

    ax1.set_title(f"Wireframe Comparison (Amplified {amp}x)")
    ax1.legend()
    ax1.axis('equal')

    # --- BOTTOM PLOT: ERROR HEATMAP ---
    # This shows WHERE the model is failing
    scatter = ax2.scatter(pr_pos[:, 0], pr_pos[:, 1], c=error, cmap='viridis', s=30)
    plt.colorbar(scatter, ax=ax2, label="Absolute Error (m)")
    
    # Draw wireframe for context
    for i in range(edges.shape[1]):
        n1, n2 = edges[0, i], edges[1, i]
        ax2.plot([pr_pos[n1,0], pr_pos[n2,0]], [pr_pos[n1,1], pr_pos[n2,1]], 'black', alpha=0.1, lw=0.5)

    ax2.set_title("Prediction Error Heatmap (Where is the AI struggling?)")
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()