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
    dataset = get_dataset(root="data/")

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
