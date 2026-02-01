import os
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import streamlit as st
import torch
from torch_geometric.data import Batch

# --- CONFIGURATION DU CHEMIN ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.fea_gnn.data_loader import PlateHoleDataset
from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config

# --- CONFIG STREAMLIT ---
st.set_page_config(layout="wide", page_title="Tests de Généralisation GNN")


@st.cache_resource
def load_model_and_config():
    """Charge le modèle entraîné une seule fois."""
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridPhysicsGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    # Tentative de chargement du modèle (standard ou scaled)
    path = os.path.join(cfg["env"]["save_path"], "gnn_hybrid_scaled.pth")
    if not os.path.exists(path):
        path = os.path.join(cfg["env"]["save_path"], "gnn_hybrid.pth")

    if os.path.exists(path):
        try:
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=True)
            )
        except RuntimeError:
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=True), strict=False
            )
        model.eval()
        return model, cfg, device, True
    else:
        return None, cfg, device, False


def load_specific_case(case_name, force_reprocess=False):
    """Charge (et re-traite si demandé) le dataset spécifique à un cas test."""
    root_dir = f"data/generalization/{case_name}"
    processed_file = os.path.join(root_dir, "processed/dataset.pt")

    # Vérification que les données brutes existent (générées par gen_test_shapes.py)
    if not os.path.exists(os.path.join(root_dir, "db.csv")):
        return None

    if force_reprocess and os.path.exists(processed_file):
        try:
            os.remove(processed_file)
        except OSError:
            pass

    # PlateHoleDataset gère la création automatique du .pt
    dataset = PlateHoleDataset(root=root_dir)
    return dataset[0] if len(dataset) > 0 else None


def plot_generalization_result(model, data, device, scale_factor, norm_cfg):
    """Génère la figure comparative et calcule les métriques."""
    # Prédiction
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        pred = model(batch).cpu().numpy()

    u_true = data.y.numpy()
    u_pred = pred
    pos = data.x[:, 0:2].numpy()

    # Dé-normalisation
    pos[:, 0] *= float(norm_cfg["x"])
    pos[:, 1] *= float(norm_cfg["y"])

    # Métriques
    error = np.linalg.norm(u_true - u_pred, axis=1)
    mae = np.mean(error)
    max_err = np.max(error)
    u_mag = np.linalg.norm(u_true, axis=1)
    mask = u_mag > 1e-3
    rel_err = (error[mask] / u_mag[mask] * 100).mean() if mask.any() else 0.0

    # Triangulation
    if hasattr(data, "face") and data.face is not None:
        triangles = data.face.t().numpy()
        triang = tri.Triangulation(pos[:, 0], pos[:, 1], triangles=triangles)
    else:
        triang = tri.Triangulation(pos[:, 0], pos[:, 1])

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Superposition
    axs[0].set_title(f"Superposition Géométrique (x{scale_factor})")
    axs[0].set_aspect("equal")
    axs[0].triplot(triang, "k-", alpha=0.05)  # Maillage initial

    pos_t = pos + (u_true / 1000.0) * scale_factor
    axs[0].triplot(
        pos_t[:, 0],
        pos_t[:, 1],
        triang.triangles,
        "b-",
        alpha=0.5,
        label="FEniCS (Vérité)",
    )

    pos_p = pos + (pred / 1000.0) * scale_factor
    axs[0].triplot(
        pos_p[:, 0],
        pos_p[:, 1],
        triang.triangles,
        "r--",
        alpha=0.8,
        label="GNN (Prédiction)",
    )
    axs[0].legend()

    # 2. Heatmap Erreur
    axs[1].set_title("Carte d'Erreur Absolue (mm)")
    axs[1].set_aspect("equal")
    tpc = axs[1].tripcolor(triang, error, cmap="inferno", shading="gouraud")
    plt.colorbar(tpc, ax=axs[1], label="Erreur (mm)")

    return fig, mae, max_err, rel_err


def main():
    st.title("🛡️ Crash-Test de Généralisation")
    st.markdown(
        """
    Cette interface teste la robustesse du modèle sur des **géométries jamais vues** lors de l'entraînement.
    Cela permet de vérifier si l'IA a appris la physique locale ou si elle a juste mémorisé la forme "Plaque à 1 trou".
    """
    )

    model, cfg, device, success = load_model_and_config()

    if not success:
        st.error("Modèle introuvable. Veuillez entraîner le modèle d'abord.")
        return

    # --- SIDEBAR ---
    st.sidebar.header("Configuration du Test")

    case_options = {
        "Plaque Pleine (Pas de trou)": "full_plate",
        "Double Trou (Inédit)": "double_hole",
    }
    selected_label = st.sidebar.radio("Cas de Test", list(case_options.keys()))
    selected_case = case_options[selected_label]

    st.sidebar.divider()
    force_refresh = st.sidebar.button("🔄 Forcer le retraitement des données")
    amp_factor = st.sidebar.slider("Amplification Visuelle", 100.0, 5000.0, 1000.0)

    # --- CHARGEMENT ---
    data_sample = load_specific_case(selected_case, force_refresh)

    if data_sample is None:
        st.warning(f"Données introuvables pour '{selected_case}'.")
        st.info("Avez-vous lancé : `python data/gen_test_shapes.py` ?")
        return

    # --- RESULTATS ---
    st.subheader(f"Résultat : {selected_label}")

    fig, mae, max_err, rel_err = plot_generalization_result(
        model, data_sample, device, amp_factor, cfg["normalization"]
    )

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (Erreur Moyenne)", f"{mae:.4f} mm")
    c2.metric("Erreur Max", f"{max_err:.4f} mm")
    c3.metric("Erreur Relative", f"{rel_err:.2f} %", delta_color="inverse")

    st.pyplot(fig)

    # Interprétation
    if rel_err < 15.0:
        st.success(
            "✅ **Succès :** Le modèle généralise bien sur cette nouvelle géométrie."
        )
    else:
        st.warning(
            "⚠️ **Attention :** Le modèle peine à prédire cette forme inconnue (Erreur > 15%)."
        )


if __name__ == "__main__":
    main()
