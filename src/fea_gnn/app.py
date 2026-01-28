import io
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import streamlit as st
import torch
from torch_geometric.data import Batch

# --- IMPORTS PROJET ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.fea_gnn.data_loader import get_dataset
from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config

# --- CONSTANTES ---
TARGET_SCALE = 1_000_000.0

st.set_page_config(layout="wide", page_title="FEA-GNN Analyseur de Déformations")


# --- FONCTIONS DE CHARGEMENT ---
@st.cache_resource
def load_data_and_model():
    cfg = load_config()
    device = torch.device("cpu")

    dataset = get_dataset(root="data/")

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
        return dataset, model, "Modèle SCALED chargé.", "success"
    elif os.path.exists(path_normal):
        model.load_state_dict(
            torch.load(path_normal, map_location=device, weights_only=True)
        )
        model.eval()
        return dataset, model, "Modèle STANDARD chargé.", "warning"
    else:
        return None, None, "Aucun modèle trouvé.", "error"


# --- FONCTIONS DE TRIANGULATION ---
def create_triangulation(pos, data_raw):
    if hasattr(data_raw, "face") and data_raw.face is not None:
        triangles = data_raw.face.t().numpy()
        return tri.Triangulation(pos[:, 0], pos[:, 1], triangles=triangles)
    else:
        return tri.Triangulation(pos[:, 0], pos[:, 1])


# --- FONCTIONS DE PLOT ---
def plot_side_by_side(pos, triangles, u_true, u_pred, title_ref, width, height, vmax):
    """
    Affiche les 3 graphiques côte à côte (1 ligne, 3 colonnes).
    Calcule la hauteur pour éviter l'écrasement.
    """
    ratio = height / width if width > 0 else 1.0

    # Largeur totale de la figure (en pouces)
    fig_width = 20
    # Hauteur nécessaire = (Largeur d'un sous-plot) * Ratio * Marge
    # Un sous-plot fait 1/3 de la largeur.
    fig_height = (fig_width / 3) * ratio * 1.3

    fig, axs = plt.subplots(
        1, 3, figsize=(fig_width, fig_height), constrained_layout=True
    )

    triang = tri.Triangulation(pos[:, 0], pos[:, 1], triangles=triangles)

    # 1. Initial
    axs[0].set_aspect("equal")
    axs[0].set_title("Maillage Initial", fontsize=14, fontweight="bold")
    axs[0].triplot(triang, "k-", alpha=0.5, linewidth=0.5)
    axs[0].axis("off")

    # 2. Reference
    axs[1].set_aspect("equal")
    axs[1].set_title(title_ref, fontsize=14, fontweight="bold")
    coll1 = axs[1].tripcolor(
        triang, u_true, cmap="viridis", shading="gouraud", vmin=0, vmax=vmax
    )
    axs[1].axis("off")

    # 3. Prediction
    axs[2].set_aspect("equal")
    axs[2].set_title("Prédiction GNN", fontsize=14, fontweight="bold")
    coll2 = axs[2].tripcolor(
        triang, u_pred, cmap="viridis", shading="gouraud", vmin=0, vmax=vmax
    )
    axs[2].axis("off")

    # Barre de couleur commune à droite
    cbar = fig.colorbar(coll2, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Déplacement (mm)", fontsize=12)

    return fig


def plot_superposition_figure(
    pos_initial, u_true, u_pred, triangles, scale_factor, width, height
):
    ratio = height / width if width > 0 else 1.0
    fig_width = 10
    fig_height = fig_width * ratio * 1.1

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    ax.set_aspect("equal")
    ax.axis("off")

    tri_obj_initial = tri.Triangulation(
        pos_initial[:, 0], pos_initial[:, 1], triangles=triangles
    )

    # 1. INITIAL (GRIS)
    ax.triplot(
        tri_obj_initial,
        color="gray",
        linestyle="--",
        alpha=0.3,
        linewidth=0.5,
        label="Initial",
        zorder=1,
    )

    # 2. REFERENCE (BLEU)
    pos_true = pos_initial + (u_true / 1000.0) * scale_factor
    tri_true = tri.Triangulation(pos_true[:, 0], pos_true[:, 1], triangles=triangles)
    ax.triplot(
        tri_true, color="blue", alpha=0.6, linewidth=1.2, label="Référence", zorder=2
    )

    # 3. GNN (ROUGE)
    pos_pred = pos_initial + (u_pred / 1000.0) * scale_factor
    tri_pred = tri.Triangulation(pos_pred[:, 0], pos_pred[:, 1], triangles=triangles)
    ax.triplot(tri_pred, color="red", alpha=0.8, linewidth=1.2, label="GNN", zorder=3)

    all_x = np.concatenate([pos_initial[:, 0], pos_true[:, 0], pos_pred[:, 0]])
    all_y = np.concatenate([pos_initial[:, 1], pos_true[:, 1], pos_pred[:, 1]])
    margin_x = (np.max(all_x) - np.min(all_x)) * 0.1
    margin_y = (np.max(all_y) - np.min(all_y)) * 0.1
    ax.set_xlim(np.min(all_x) - margin_x, np.max(all_x) + margin_x)
    ax.set_ylim(np.min(all_y) - margin_y, np.max(all_y) + margin_y)

    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(
        f"Visualisation (Amplification x{scale_factor})", fontsize=14, fontweight="bold"
    )
    return fig


def plot_error_analysis(
    pos, triangles, error_mm, mag_true, mag_pred, width, height, vmax
):
    """Affiche l'erreur et la corrélation côte à côte."""
    ratio = height / width if width > 0 else 1.0
    fig_width = 16
    fig_height = (fig_width / 2) * ratio * 1.2  # Hauteur adaptée

    fig, axs = plt.subplots(
        1, 2, figsize=(fig_width, fig_height), constrained_layout=True
    )

    triang = tri.Triangulation(pos[:, 0], pos[:, 1], triangles=triangles)

    # Carte Erreur
    axs[0].set_aspect("equal")
    axs[0].set_title("Distribution de l'Erreur", fontsize=14, fontweight="bold")
    coll = axs[0].tripcolor(triang, error_mm, cmap="inferno", shading="gouraud")
    axs[0].axis("off")
    cbar = fig.colorbar(coll, ax=axs[0], fraction=0.046, pad=0.04)
    cbar.set_label("Erreur (mm)", fontsize=10)

    # Scatter Plot
    axs[1].set_aspect("equal")  # Carré pour la corrélation
    axs[1].set_title("Corrélation", fontsize=14, fontweight="bold")
    axs[1].scatter(mag_true, mag_pred, alpha=0.1, s=10, c="blue")
    axs[1].plot([0, vmax], [0, vmax], "r--", linewidth=2)
    axs[1].set_xlabel("Référence (mm)")
    axs[1].set_ylabel("GNN (mm)")
    axs[1].grid(True, alpha=0.3)

    return fig


# --- MAIN ---
def main():
    st.title("FEA-GNN : Analyseur de Déformations")

    dataset, model, msg, status = load_data_and_model()

    if status == "error":
        st.error(msg)
        return
    elif status == "success" or status == "warning":
        st.toast(msg)

    if dataset is None:
        return

    # --- SIDEBAR ---
    st.sidebar.header("Contrôle Simulation")
    mode = st.sidebar.radio(
        "Mode", ["Validation (Comparaison)", "Stress Test (Extremums)"], index=0
    )
    st.sidebar.divider()

    sim_index = st.sidebar.number_input("ID Simulation", 0, len(dataset) - 1, 0)
    data_raw = dataset[sim_index]

    # --- PROPRIETES ---
    pos_initial = data_raw.x[:, 0:2].numpy()
    x_min, x_max = np.min(pos_initial[:, 0]), np.max(pos_initial[:, 0])
    y_min, y_max = np.min(pos_initial[:, 1]), np.max(pos_initial[:, 1])
    width = x_max - x_min
    height = y_max - y_min

    st.sidebar.markdown("### Propriétés")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Largeur (m)", f"{width:.3f}")
    c2.metric("Hauteur (m)", f"{height:.3f}")

    E_val = data_raw.x[0, 2].item()
    nu_val = data_raw.x[0, 3].item()
    c3, c4 = st.sidebar.columns(2)
    c3.metric("Young (E)", f"{E_val:.2e}")
    c4.metric("Poisson", f"{nu_val:.2f}")

    # --- INFERENCE ---
    factor_mm = 1000.0 / TARGET_SCALE

    if mode == "Validation (Comparaison)":
        batch = Batch.from_data_list([data_raw])
        with torch.no_grad():
            pred = model(batch)
        u_true = data_raw.y.numpy() * factor_mm
        u_pred = pred.numpy() * factor_mm
        title_ref = "Vérité Terrain (FEniCS)"
    else:
        st.sidebar.markdown("### Stress Test")
        force_mult = st.sidebar.slider("Multiplicateur Force", 0.0, 20.0, 1.0, 0.5)
        data_mod = data_raw.clone()
        data_mod.x[:, 4:6] *= force_mult
        batch = Batch.from_data_list([data_mod])
        with torch.no_grad():
            pred = model(batch)
        u_pred = pred.numpy() * factor_mm
        u_true = (data_raw.y.numpy() * factor_mm) * force_mult
        title_ref = f"Théorie Linéaire (x{force_mult})"

    mag_true = np.linalg.norm(u_true, axis=1)
    mag_pred = np.linalg.norm(u_pred, axis=1)
    error_mm = np.linalg.norm(u_true - u_pred, axis=1)

    # --- METRIQUES ---
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ref Max (mm)", f"{np.max(mag_true):.3f}")
    m2.metric("GNN Max (mm)", f"{np.max(mag_pred):.3f}")
    m3.metric("Erreur Moy.", f"{np.mean(error_mm):.4f}")
    m4.metric("Erreur Max", f"{np.max(error_mm):.4f}")

    # Triangulation
    tri_obj = create_triangulation(pos_initial, data_raw)
    triangles = tri_obj.triangles
    vmax = max(np.max(mag_true), np.max(mag_pred))
    if vmax == 0:
        vmax = 1e-6

    # --- 1. COMPARAISON VISUELLE (COTE A COTE) ---
    st.divider()
    st.header("1. Comparaison Visuelle des Champs")

    # Appel de la nouvelle fonction qui plotte tout sur une ligne
    fig_compare = plot_side_by_side(
        pos_initial, triangles, mag_true, mag_pred, title_ref, width, height, vmax
    )
    st.pyplot(fig_compare, use_container_width=True)

    # --- 2. ERREUR ---
    st.divider()
    st.header("2. Analyse de l'Erreur")
    fig_error = plot_error_analysis(
        pos_initial, triangles, error_mm, mag_true, mag_pred, width, height, vmax
    )
    st.pyplot(fig_error, use_container_width=True)

    # --- 3. SUPERPOSITION ---
    st.divider()
    st.header("3. Superposition Géométrique")

    _, col_slider, _ = st.columns([1, 2, 1])
    with col_slider:
        scale_factor = st.slider("Amplification Visuelle", 1.0, 100.0, 20.0)

    col_l, col_c, col_r = st.columns([1, 6, 1])
    with col_c:
        fig_ov = plot_superposition_figure(
            pos_initial, u_true, u_pred, triangles, scale_factor, width, height
        )
        st.pyplot(fig_ov, use_container_width=True)


if __name__ == "__main__":
    main()
