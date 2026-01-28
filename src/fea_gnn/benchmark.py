import argparse
import os
import sys
import time
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
# --- IMPORTS SÉCURISÉS ---
# On ne met QUE les imports standards ici.
# Les imports lourds (torch, dolfinx) sont déplacés dans les fonctions.


def benchmark_gnn(device_name="cuda"):
    """Mesure le temps d'inférence moyen du modèle GNN."""
    print(f"\n--- 1. Benchmark GNN ({device_name}) ---")

    # Imports locaux pour éviter de bloquer le mode FEniCS
    try:
        import torch

        from src.fea_gnn.data_loader import get_dataset
        from src.fea_gnn.model import HybridPhysicsGNN
        from src.fea_gnn.utils import load_config
    except ImportError:
        print("Erreur : PyTorch ou les modules locaux sont introuvables.")
        print("Lancez ce mode avec 'uv run python -m ...'")
        return None

    cfg = load_config()
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(root="data/")
    if len(dataset) == 0:
        print("Erreur: Dataset vide.")
        return 0.0

    data = dataset[0].to(device)
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    model = HybridPhysicsGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    print("Chauffe du GPU...")
    for _ in range(10):
        _ = model(data)

    n_loops = 1000
    print(f"Lancement de {n_loops} inférences...")

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(n_loops):
            _ = model(data)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / n_loops
    print(f"Temps moyen GNN : {avg_time * 1000:.4f} ms")
    return avg_time


def benchmark_fenics():
    """Mesure le temps de résolution par Éléments Finis."""
    print("\n--- 2. Benchmark FEniCS (CPU) ---")

    try:
        import dolfinx
        from mpi4py import MPI

        # Import dynamique des fonctions de ton générateur de données
        from data.data_generator_2d import (create_plate_with_hole_mesh,
                                            solve_elasticity)
    except ImportError as e:
        print(f"FEniCS non disponible : {e}")
        return None

    comm = MPI.COMM_WORLD
    L, H, R = 2.0, 0.5, 0.1
    E, nu, F = 210e9, 0.3, 1e6

    times = []
    n_loops = 5
    print(f"Lancement de {n_loops} simulations FEA complètes...")

    for i in range(n_loops):
        t0 = time.time()
        msh = create_plate_with_hole_mesh(comm, L, H, R, res_factor=12)
        _ = solve_elasticity(msh, E, nu, F)
        t1 = time.time()
        times.append(t1 - t0)
        print(f"  Simulation {i + 1}: {t1 - t0:.4f} s")

    avg_time = sum(times) / len(times)
    print(f"Temps moyen FEniCS : {avg_time:.4f} s")
    return avg_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gnn", "fenics"], required=True)
    args = parser.parse_args()

    if args.mode == "gnn":
        benchmark_gnn()
    elif args.mode == "fenics":
        benchmark_fenics()
