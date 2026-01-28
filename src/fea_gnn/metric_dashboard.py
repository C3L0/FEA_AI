from os import wait

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from scipy.stats import binned_statistic_2d
from torch_geometric.loader import DataLoader

from src.fea_gnn.data_loader import get_dataset
from src.fea_gnn.model import HybridPhysicsGNN
# Imports internes
from src.fea_gnn.utils import load_config

# Configuration de la page
st.set_page_config(page_title="FEA-GNN Dashboard", layout="wide")


# --- CHARGEMENT DES RESSOURCES (CACHÉ) ---
@st.cache_resource
def load_resources():
    cfg = load_config()
    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = get_dataset(root="data/")

    # Modèle
    model = HybridPhysicsGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    # Chargement des poids
    path = f"{cfg['env']['save_path']}/gnn_hybrid_scaled.pth"
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
    except FileNotFoundError:
        st.error(f"Modèle introuvable : {path}")
        return None, None, None, None

    return model, dataset, cfg, device


@st.cache_data
def load_history(cfg):
    try:
        return pd.read_csv(f"{cfg['env']['save_path']}/training_history.csv")
    except FileNotFoundError:
        return None


# --- FONCTION D'INFERENCE EN MASSE ---
@st.cache_data
def run_batch_inference(_model, _dataset, _cfg, _device, num_samples=100):
    """Exécute le modèle sur N échantillons et extrait les métriques et paramètres."""
    loader = DataLoader(_dataset, batch_size=1, shuffle=True)
    results = []

    # Stockage pour la heatmap globale
    all_x, all_y, all_errors = [], [], []

    norm = _cfg["normalization"]
    count = 0

    with torch.no_grad():
        for data in loader:
            if count >= num_samples:
                break
            data = data.to(_device)
            pred = _model(data)

            # --- CALCUL D'ERREUR ---
            # Unités réelles (mm)
            target_mm = data.y.cpu().numpy()
            pred_mm = pred.cpu().numpy()

            mae = np.mean(np.abs(pred_mm - target_mm))
            mse = np.mean((pred_mm - target_mm) ** 2)
            max_err = np.max(np.linalg.norm(pred_mm - target_mm, axis=1))

            # --- EXTRACTION PARAMÈTRES PHYSIQUES ---
            # data.x : [x, y, E, nu, Fx, Fy, isFixed]
            # On prend la moyenne des features sur le graphe car E, nu sont constants
            E_gpa = data.x[:, 2].mean().item() * float(norm["E"]) / 1e9
            nu = data.x[:, 3].mean().item() * float(norm["nu"])

            # Force : on regarde les nœuds chargés (Fx != 0)
            fx = data.x[:, 4] * float(norm["force"])
            fy = data.x[:, 5] * float(norm["force"])
            force_mag = torch.max(torch.sqrt(fx**2 + fy**2)).item()

            num_nodes = data.num_nodes

            results.append(
                {
                    "ID": int(data.sim_id[0]),
                    "MAE (mm)": mae,
                    "MSE (mm²)": mse,
                    "Max Error (mm)": max_err,
                    "Young (GPa)": E_gpa,
                    "Poisson": nu,
                    "Force (N)": force_mag,
                    "Noeuds": num_nodes,
                }
            )

            # --- DONNÉES SPATIALES POUR HEATMAP ---
            # Coordonnées dé-normalisées (mètres)
            coords = data.x[:, 0:2].cpu().numpy()
            coords[:, 0] *= float(norm["x"])
            coords[:, 1] *= float(norm["y"])

            # Erreur locale (norme euclidienne)
            local_err = np.linalg.norm(pred_mm - target_mm, axis=1)

            all_x.append(coords[:, 0])
            all_y.append(coords[:, 1])
            all_errors.append(local_err)

            count += 1

    return (
        pd.DataFrame(results),
        np.concatenate(all_x),
        np.concatenate(all_y),
        np.concatenate(all_errors),
    )


# --- INTERFACE UTILISATEUR ---

st.title("Dashboard d'Analyse GNN (Plaques Trouées)")

model, dataset, cfg, device = load_resources()

if model is not None:
    # Sidebar
    st.sidebar.header("Paramètres d'Analyse")
    num_samples = st.sidebar.slider("Nombre de simulations à tester", 10, 500, 50)

    # Chargement des données
    with st.spinner(f"Inférence sur {num_samples} simulations..."):
        df_results, x_flat, y_flat, err_flat = run_batch_inference(
            model, dataset, cfg, device, num_samples
        )

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(
        ["Entraînement", "Analyse Paramétrique", "Erreur Spatiale"]
    )

    # --- TAB 1: ENTRAÎNEMENT ---
    with tab1:
        history = load_history(cfg)
        if history is not None:
            st.markdown("### Convergence du Modèle")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=history["epoch"],
                    y=history["total_loss"],
                    name="Total Loss",
                    line=dict(color="black"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=history["epoch"],
                    y=history["data_loss"],
                    name="Data Loss (MAE)",
                    line=dict(dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=history["epoch"],
                    y=history["phys_loss"],
                    name="Physics Loss",
                    line=dict(dash="dot"),
                )
            )

            fig.update_layout(
                yaxis_type="log",
                title="Évolution des Loss (Échelle Log)",
                xaxis_title="Époque",
                yaxis_title="Valeur",
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            col1.metric("Dernière Data Loss", f"{history['data_loss'].iloc[-1]:.6f}")
            col1.metric("Dernière Phys Loss", f"{history['phys_loss'].iloc[-1]:.6f}")
        else:
            st.warning("Pas d'historique d'entraînement trouvé.")

    # --- TAB 2: CORRÉLATIONS ---
    with tab2:
        st.markdown("### Influence des Paramètres Physiques sur la Précision")
        st.dataframe(df_results.describe())

        col_x = st.selectbox(
            "Paramètre X (Axe horizontal)",
            ["Young (GPa)", "Poisson", "Force (N)", "Noeuds"],
            index=0,
        )
        col_y = st.selectbox(
            "Métrique Y (Axe vertical)",
            ["MAE (mm)", "Max Error (mm)", "MSE (mm²)"],
            index=0,
        )

        # Scatter Plot avec couleur pour une 3ème dimension
        fig = px.scatter(
            df_results,
            x=col_x,
            y=col_y,
            color="Force (N)" if col_x != "Force (N)" else "Young (GPa)",
            size="Noeuds",
            hover_data=["ID"],
            title=f"Corrélation : {col_y} vs {col_x}",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Analyse :** Si les points montent vers la droite, cela signifie que le modèle a plus de mal avec les valeurs élevées de ce paramètre."
        )

    # --- TAB 3: HEATMAP SPATIALE ---
    with tab3:
        st.markdown("### Carte Moyenne des Erreurs (Spatial Binning)")
        st.write(
            f"Cette carte agrège les erreurs de **{num_samples} formes différentes** en divisant l'espace en une grille régulière."
        )

        # Paramètres du Binning
        bins_x = 50
        bins_y = 20

        # Calcul de la statistique bivariée (Moyenne de l'erreur dans chaque case)
        # On utilise les coordonnées réelles (m)
        ret = binned_statistic_2d(
            x_flat,
            y_flat,
            err_flat,
            statistic="mean",
            bins=[bins_x, bins_y],
            range=[[0, 2.5], [-0.6, 0.6]],  # Bornes approximatives de la plaque
        )

        # Affichage avec Matplotlib
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            ret.statistic.T,
            origin="lower",
            extent=[0, 2.5, -0.6, 0.6],
            cmap="hot_r",  # Blanc = Erreur faible, Rouge/Noir = Erreur forte
            aspect="equal",
        )
        plt.colorbar(im, label="Erreur Moyenne Absolue (mm)")
        plt.title(
            f"Distribution Spatiale de l'Erreur (Moyenne sur {num_samples} simulations)"
        )
        plt.xlabel("Position X (m)")
        plt.ylabel("Position Y (m)")
        st.pyplot(fig)

        st.warning(
            "**Note :** Les zones blanches vides correspondent aux endroits où il n'y a jamais de matière (le trou central variable)."
        )
