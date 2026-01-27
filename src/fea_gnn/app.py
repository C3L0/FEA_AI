import os
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import streamlit as st
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

# --- IMPORTS PROJET ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.fea_gnn.data_loader import get_dataset
from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config

# --- CONSTANTES ---
TARGET_SCALE = 1_000_000.0

st.set_page_config(layout="wide", page_title="FEA-GNN Visualizer")


# --- FONCTIONS DE CHARGEMENT (PURE) ---
@st.cache_resource
def load_data_and_model():
    """
    Charge le dataset et le modèle.
    Ne contient AUCUN appel Streamlit (st.write, st.toast, etc.) pour éviter l'erreur de cache.
    Renvoie : dataset, model, message_status, type_status
    """
    cfg = load_config()
    device = torch.device("cpu")

    # 1. Dataset
    dataset = get_dataset(root="data/")

    # 2. Modèle
    model = HybridPhysicsGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    path_scaled = os.path.join(cfg["env"]["save_path"], "gnn_hybrid_scaled.pth")
    path_normal = os.path.join(cfg["env"]["save_path"], "gnn_hybrid.pth")

    if os.path.exists(path_scaled):
        model.load_state_dict(
            torch.load(path_scaled, map_location=device, weights_only=True)
        )
        model.eval()
        return dataset, model, f"Modèle SCALED chargé : {path_scaled}", "success"

    elif os.path.exists(path_normal):
        model.load_state_dict(
            torch.load(path_normal, map_location=device, weights_only=True)
        )
        model.eval()
        return dataset, model, f"Modèle STANDARD chargé : {path_normal}", "warning"

    else:
        return None, None, "Aucun modèle trouvé ! Entraîne d'abord le modèle.", "error"


# --- FONCTIONS DE PLOT ---
def create_triangulation(pos, edge_index):
    return tri.Triangulation(pos[:, 0], pos[:, 1])


def plot_field_on_mesh(ax, triang, values, title, cmap="viridis", vmin=None, vmax=None):
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")
    collection = ax.tripcolor(
        triang, values, cmap=cmap, shading="gouraud", vmin=vmin, vmax=vmax
    )
    ax.triplot(triang, "k-", alpha=0.1, linewidth=0.5)
    ax.axis("off")
    return collection


# --- INTERFACE PRINCIPALE ---
def main():
    st.title("🔬 FEA-GNN : Analyseur de Déformations")
    st.markdown(
        "Visualisation comparative entre simulation FEniCS (Vérité Terrain) et Prédiction GNN."
    )

    # 1. Chargement (Fonction pure)
    dataset, model, status_msg, status_type = load_data_and_model()

    # 2. Affichage des statuts (Hors du cache)
    if status_type == "error":
        st.error(status_msg)
        return
    elif status_type == "warning":
        st.toast(status_msg, icon="⚠️")
    elif status_type == "success":
        st.toast(status_msg, icon="✅")

    if dataset is None:
        return

    # --- SIDEBAR ---
    st.sidebar.header("Paramètres")
    sim_index = st.sidebar.number_input(
        "Index Simulation", min_value=0, max_value=len(dataset) - 1, value=0, step=1
    )

    # Récupération des données
    data_raw = dataset[sim_index]

    # Création du Batch pour l'inférence
    data_batch = Batch.from_data_list([data_raw])

    # INFÉRENCE
    with torch.no_grad():
        pred_scaled = model(data_batch)

    # --- DÉSÉCHELONNAGE ---
    pos_initial = data_raw.x[:, 0:2].numpy()
    factor_mm = 1000.0 / TARGET_SCALE

    u_true_mm = data_raw.y.numpy() * factor_mm
    u_pred_mm = pred_scaled.numpy() * factor_mm

    mag_true = np.linalg.norm(u_true_mm, axis=1)
    mag_pred = np.linalg.norm(u_pred_mm, axis=1)
    error_mm = np.linalg.norm(u_true_mm - u_pred_mm, axis=1)

    # --- MÉTRIQUES CLÉS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Déplacement Max (Vrai)", f"{np.max(mag_true):.3f} mm")
    col2.metric(
        "Déplacement Max (IA)",
        f"{np.max(mag_pred):.3f} mm",
        delta=f"{np.max(mag_pred) - np.max(mag_true):.3f} mm",
    )
    col3.metric("Erreur Moyenne (MAE)", f"{np.mean(error_mm):.4f} mm")
    col4.metric("Erreur Max", f"{np.max(error_mm):.3f} mm")

    st.divider()

    # --- LIGNE 1 : LES FORMES ---
    st.subheader("1. Comparaison Visuelle des Champs de Déplacement")
    triang = create_triangulation(pos_initial, data_raw.edge_index)

    fig1, axs = plt.subplots(1, 3, figsize=(18, 5))
    vmax = max(np.max(mag_true), np.max(mag_pred))

    axs[0].set_title("Forme Initiale", fontweight="bold")
    axs[0].triplot(triang, "k-", alpha=0.3, linewidth=0.5)
    axs[0].set_aspect("equal")
    axs[0].axis("off")

    im2 = plot_field_on_mesh(
        axs[1],
        triang,
        mag_true,
        "Cible FEniCS (Vérité)",
        cmap="viridis",
        vmin=0,
        vmax=vmax,
    )
    im3 = plot_field_on_mesh(
        axs[2], triang, mag_pred, "Prédiction GNN", cmap="viridis", vmin=0, vmax=vmax
    )

    cbar = fig1.colorbar(im3, ax=axs, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Déplacement (mm)")
    st.pyplot(fig1)

    # --- LIGNE 2 : L'ERREUR ---
    st.subheader("2. Où le modèle se trompe-t-il ?")
    col_err_1, col_err_2 = st.columns([2, 1])

    with col_err_1:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        im_err = plot_field_on_mesh(
            ax2, triang, error_mm, "Erreur Absolue (mm)", cmap="inferno"
        )
        cbar2 = plt.colorbar(im_err, ax=ax2)
        st.pyplot(fig2)

    with col_err_2:
        st.markdown("**Analyse de Parité**")
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        ax3.scatter(mag_true, mag_pred, alpha=0.1, s=5, c="blue")
        max_val = max(np.max(mag_true), np.max(mag_pred))
        ax3.plot([0, max_val], [0, max_val], "r--", label="Idéal")
        ax3.set_xlabel("Vrai (mm)")
        ax3.set_ylabel("Prédit (mm)")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

    # --- LIGNE 3 : SUPERPOSITION ---
    st.subheader("3. Superposition Géométrique (Amplifiée)")
    scale_factor = st.slider("Facteur d'amplification", 1.0, 50.0, 10.0)

    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.set_aspect("equal")
    ax4.axis("off")

    ax4.triplot(triang, "k--", alpha=0.1, linewidth=0.5, label="Initial")

    pos_true = pos_initial + (data_raw.y.numpy() / TARGET_SCALE * 1000.0) * scale_factor
    triang_true = tri.Triangulation(pos_true[:, 0], pos_true[:, 1], triang.triangles)
    ax4.triplot(triang_true, "b-", alpha=0.4, linewidth=0.8, label="FEniCS")

    pos_pred = (
        pos_initial + (pred_scaled.numpy() / TARGET_SCALE * 1000.0) * scale_factor
    )
    triang_pred = tri.Triangulation(pos_pred[:, 0], pos_pred[:, 1], triang.triangles)
    ax4.triplot(triang_pred, "r-", alpha=0.6, linewidth=1.0, label="GNN")

    ax4.legend()
    st.pyplot(fig4)


if __name__ == "__main__":
    main()
