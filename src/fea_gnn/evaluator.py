import os

import numpy as np
import torch
import yaml
# Utilisation du DataLoader spécifique aux graphes
from torch_geometric.loader import DataLoader

# Imports internes corrigés
from src.fea_gnn.data_loader import get_dataset  # Le bon loader
from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config
from src.fea_gnn.visualizer import plot_fea_comparison


def compute_metrics(pred_tensor, target_tensor, scale_factor=1000.0):
    """Calcule les métriques sur CPU."""
    pred_m = pred_tensor.numpy() / scale_factor
    target_m = target_tensor.numpy() / scale_factor

    mse = np.mean((pred_m - target_m) ** 2)
    mae = np.mean(np.abs(pred_m - target_m))

    # Erreur max (norme euclidienne par nœud)
    node_err = np.linalg.norm(pred_m - target_m, axis=1)
    max_error = np.max(node_err)

    return {"mse": mse, "mae": mae, "max_error": max_error}


def evaluate_on_test_set(model, dataset, device, num_samples=50):
    model.eval()
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False
    )  # Batch size petit pour eval

    agg_metrics = {"mse": [], "mae": [], "max_error": []}
    count = 0

    print(f"[Evaluator] Analyse quantitative sur {num_samples} échantillons...")

    with torch.no_grad():
        for data in loader:
            if count >= num_samples:
                break

            data = data.to(device)

            # Prédiction
            pred = model(data)

            # Calcul métriques (data.y est la cible)
            batch_metrics = compute_metrics(pred.cpu(), data.y.cpu())

            for k, v in batch_metrics.items():
                agg_metrics[k].append(v)

            count += data.num_graphs

    return {k: np.mean(v) for k, v in agg_metrics.items()}


def run_evaluation_pipeline():
    cfg = load_config()
    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")

    # 1. Chargement des données
    print("[Evaluator] Chargement du dataset...")
    # On utilise get_dataset() qui pointe vers processed/dataset.pt
    dataset = get_dataset()

    # 2. Chargement du modèle
    print("[Evaluator] Initialisation du modèle...")
    model = HybridPhysicsGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    model_path = os.path.join(cfg["env"]["save_path"], "gnn_hybrid.pth")
    if not os.path.exists(model_path):
        print(f"ERREUR: Modèle introuvable à {model_path}")
        return

    # Chargement des poids
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("[Evaluator] Poids chargés.")

    # 3. Évaluation Quantitative
    metrics = evaluate_on_test_set(model, dataset, device)
    print("\n" + "=" * 40)
    print("   RAPPORT D'ÉVALUATION (GNN Hybride)")
    print("=" * 40)
    print(f" MSE (Erreur Quadratique):  {metrics['mse']:.2e} m²")
    print(f" MAE (Erreur Absolue):      {metrics['mae']:.2e} m")
    print(f" Erreur Max (Pire Cas):     {metrics['max_error']:.2e} m")
    print("=" * 40 + "\n")

    # 4. Visualisation Qualitative
    print("[Evaluator] Génération des graphiques...")
    # On prend quelques indices aléatoires
    indices = [0, 10, 20] if len(dataset) > 20 else [0]
    for i in indices:
        plot_fea_comparison(model, dataset, device, sample_idx=i)


if __name__ == "__main__":
    run_evaluation_pipeline()

# import os
#
# import numpy as np
# import torch
# import yaml
# from torch.utils.data import DataLoader
#
# from src.fea_gnn.data_loader import CantileverMeshDataset
# # Internal Imports (Adjust depending on your exact folder structure)
# from src.fea_gnn.model import HybridPhysicsGNN
# from src.fea_gnn.visualizer import plot_fea_comparison
#
#
# def load_config(config_path="config.yaml"):
#     with open(config_path, "r") as file:
#         return yaml.safe_load(file)
#
#
# def compute_metrics(pred_batch, target_batch, scale_factor=1000.0):
#     """
#     Calculates engineering metrics.
#     Input tensors are expected to be on CPU.
#     """
#     # Unscale to Meters
#     pred_m = pred_batch.numpy() / scale_factor
#     target_m = target_batch.numpy() / scale_factor
#
#     # 1. MSE (Mean Squared Error) in Meters
#     mse = np.mean((pred_m - target_m) ** 2)
#
#     # 2. MAE (Mean Absolute Error) in Meters
#     mae = np.mean(np.abs(pred_m - target_m))
#
#     # 3. Max Error (Worst case node)
#     # Calculate euclidean distance error for each node, then max over batch
#     node_errors = np.linalg.norm(pred_m - target_m, axis=2)  # [Batch, Nodes]
#     max_error = np.max(node_errors)
#
#     return {"mse": mse, "mae": mae, "max_error": max_error}
#
#
# def evaluate_on_test_set(model, dataset, device, num_samples=100):
#     """
#     Runs inference on a subset of data and returns average metrics.
#     """
#     model.eval()
#
#     # Create a small loader for evaluation
#     loader = DataLoader(dataset, batch_size=32, shuffle=False)
#
#     agg_metrics = {"mse": [], "mae": [], "max_error": []}
#     count = 0
#
#     print(f"[Evaluator] Running quantitative evaluation on {num_samples} samples...")
#
#     with torch.no_grad():
#         for batch_x, batch_y in loader:
#             if count >= num_samples:
#                 break
#
#             batch_x = batch_x.to(device)
#
#             # Predict
#             pred = model(batch_x)
#
#             # Compute Metrics (Move to CPU for numpy)
#             batch_metrics = compute_metrics(pred.cpu(), batch_y.cpu())
#
#             for k, v in batch_metrics.items():
#                 agg_metrics[k].append(v)
#
#             count += batch_x.shape[0]
#
#     # Average everything
#     final_metrics = {k: np.mean(v) for k, v in agg_metrics.items()}
#     return final_metrics
#
#
# def run_evaluation_pipeline():
#     # 1. Configuration
#     cfg = load_config()
#     device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")
#
#     # 2. Load Data (Evaluation set)
#     print("[Evaluator] Loading Dataset...")
#     dataset = CantileverMeshDataset(
#         num_samples=200,  # Generate fresh samples for testing
#         nx=cfg["geometry"]["nx"],
#         ny=cfg["geometry"]["ny"],
#         length=cfg["geometry"]["length"],
#         height=cfg["geometry"]["height"],
#         E_range=cfg["material"]["youngs_modulus_range"],
#         nu_range=cfg["material"]["poissons_ratio_range"],
#     )
#
#     # 3. Load Model
#     print("[Evaluator] Initializing Model...")
#     model = HybridPhysicsGNN(
#         edge_index=dataset.edge_index,
#         edge_attr=dataset.edge_attr,
#         hidden_dim=cfg["model"]["hidden_dim"],
#         layers=cfg["model"]["layers"],
#         input_dim=cfg["model"]["input_dim"],
#     ).to(device)
#
#     model_path = os.path.join(cfg["env"]["save_path"], "gnn_hybrid.pth")
#     if not os.path.exists(model_path):
#         print(f"CRITICAL: No model found at {model_path}")
#         return
#
#     state_dict = torch.load(model_path, map_location=device, weights_only=True)
#     model.load_state_dict(state_dict)
#     print("[Evaluator] Model weights loaded successfully.")
#
#     # 4. Quantitative Evaluation (The Numbers)
#     metrics = evaluate_on_test_set(model, dataset, device)
#     print("\n" + "=" * 30)
#     print("   EVALUATION REPORT")
#     print("=" * 30)
#     print(f" MSE (Mean Squared Error):  {metrics['mse']:.2e} m²")
#     print(f" MAE (Mean Absolute Error): {metrics['mae']:.2e} m")
#     print(f" Worst Case Error:          {metrics['max_error']:.2e} m")
#     print("=" * 30 + "\n")
#
#     # 5. Qualitative Evaluation (The Visuals)
#     # Plot 3 different samples to check consistency
#     print("[Evaluator] Generating plots for random samples...")
#     for i in [0, 5, 10]:
#         plot_fea_comparison(model, dataset, device, sample_idx=i)
#
#
# if __name__ == "__main__":
#     run_evaluation_pipeline()
