import os

import numpy as np
import torch
import yaml
# Utilisation du DataLoader spécifique aux graphes
from torch_geometric.loader import DataLoader

# Imports internes
from src.fea_gnn.data_loader import get_dataset
from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config
from src.fea_gnn.visualizer import plot_fea_comparison

# --- CONSTANTE DE DÉSÉCHELONNAGE (REVERSE SCALING) ---
# Doit correspondre exactement au scaling utilisé dans data_pipeline.py et trainer.py
# 1e6 signifie que les données brutes sont en micromètres, on divise pour revenir en mètres.
TARGET_SCALE = 1_000_000.0


def compute_metrics(pred_tensor, target_tensor):
    """
    Calcule les métriques physiques sur CPU.
    Reçoit des données scalées (ex: 46.25) et les convertit en mètres (ex: 0.000046).
    """
    # Inverse Scaling : On revient à la réalité physique (Mètres)
    pred_m = pred_tensor.numpy() / TARGET_SCALE
    target_m = target_tensor.numpy() / TARGET_SCALE

    mse = np.mean((pred_m - target_m) ** 2)
    mae = np.mean(np.abs(pred_m - target_m))

    # Erreur max (norme euclidienne par nœud)
    # axis=1 car shape est [N_nodes, 2] (ux, uy)
    node_err = np.linalg.norm(pred_m - target_m, axis=1)
    max_error = np.max(node_err)

    return {"mse": mse, "mae": mae, "max_error": max_error}


def evaluate_on_test_set(model, dataset, device, num_samples=50):
    model.eval()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    agg_metrics = {"mse": [], "mae": [], "max_error": []}
    count = 0

    print(
        f"[Evaluator] Analyse quantitative sur {num_samples} échantillons (Échelle: Mètres)..."
    )

    with torch.no_grad():
        for data in loader:
            if count >= num_samples:
                break

            data = data.to(device)

            # Prédiction brute (Scalée x1e6)
            pred = model(data)

            # Calcul métriques (avec conversion interne en mètres)
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
    dataset = get_dataset(root="data/")

    # 2. Chargement du modèle
    print("[Evaluator] Initialisation du modèle...")
    model = HybridPhysicsGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    # MODIFICATION : On cherche le fichier 'scaled'
    model_path = os.path.join(cfg["env"]["save_path"], "gnn_hybrid_scaled.pth")

    if not os.path.exists(model_path):
        # Fallback si l'utilisateur n'a pas encore le modèle scaled
        fallback_path = os.path.join(cfg["env"]["save_path"], "gnn_hybrid.pth")
        if os.path.exists(fallback_path):
            print(
                f"ATTENTION: 'gnn_hybrid_scaled.pth' introuvable. Utilisation de '{fallback_path}'"
            )
            model_path = fallback_path
        else:
            print(f"ERREUR FATALE: Aucun modèle trouvé à {model_path}")
            return

    # Chargement des poids avec gestion d'erreurs
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[Evaluator] Poids chargés depuis : {os.path.basename(model_path)}")
    except Exception as e:
        print(f"ERREUR de chargement des poids : {e}")
        print(
            "Conseil : L'architecture du modèle dans model.py ne correspond pas au fichier .pth."
        )
        return

    # 3. Évaluation Quantitative
    metrics = evaluate_on_test_set(model, dataset, device)

    print("\n" + "=" * 50)
    print("    RAPPORT D'ÉVALUATION (GNN Hybride)")
    print("=" * 50)
    print(f" MSE (Erreur Quadratique):  {metrics['mse']:.4e} m²")
    print(
        f" MAE (Erreur Absolue):      {metrics['mae']:.4e} m ({metrics['mae'] * 1000:.2f} mm)"
    )
    print(
        f" Erreur Max (Pire Cas):     {metrics['max_error']:.4e} m ({metrics['max_error'] * 1000:.2f} mm)"
    )
    print("=" * 50 + "\n")

    # 4. Visualisation Qualitative
    print("[Evaluator] Génération des graphiques...")
    indices = [0, 10, 20] if len(dataset) > 20 else [0]

    for i in indices:
        try:
            # Note: plot_fea_comparison doit gérer le scaling en interne s'il veut afficher des mm
            # S'il utilise directement le modèle, les légendes seront x1e6 trop grandes,
            # mais la forme (rouge/bleu) sera correcte.
            plot_fea_comparison(model, dataset, device, sample_idx=i)
        except Exception as e:
            print(f"Erreur lors de la visualisation de l'index {i}: {e}")


if __name__ == "__main__":
    run_evaluation_pipeline()
