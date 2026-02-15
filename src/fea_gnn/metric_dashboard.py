import io
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from scipy.stats import binned_statistic_2d
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.fea_gnn.data_loader import get_dataset
from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config

st.set_page_config(layout="wide", page_title="FEA-GNN Analytics")


@st.cache_resource
def load_resources():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset Loading
    dataset = get_dataset(root="data/")

    # Model loading
    model = HybridPhysicsGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    path = os.path.join(cfg["env"]["save_path"], "gnn_hybrid_scaled.pth")

    if os.path.exists(path):
        try:
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=True)
            )
            status = "Model successfully loaded"
            status_type = "success"
        except RuntimeError:
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=True), strict=False
            )
            status = "Warning : Issues with the architecture, WEights load in non-strict mode"
            status_type = "warning"

        model.eval()
    else:
        status = f"Error: Model doesn't exit in the following path: {path}"
        model = None
        status_type = "error"

    return dataset, model, cfg, device, status, status_type


@st.cache_data
def load_training_history(cfg):
    path = os.path.join(cfg["env"]["save_path"], "training_history.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def run_batch_analysis_v3(_model, _dataset, _cfg, _device, num_samples=50):
    """
    Runs the model on multiple sample to extract global statistics"
    """
    loader = DataLoader(_dataset, batch_size=1, shuffle=True)
    norm = _cfg["normalization"]

    sim_stats = []
    all_x, all_y, all_error_abs, all_error_rel = [], [], [], []

    count = 0
    progress_bar = st.progress(0)

    with torch.no_grad():
        for i, data in enumerate(loader):
            if count >= num_samples:
                break

            data = data.to(_device)
            pred = _model(data)

            u_true = data.y.cpu().numpy()
            u_pred = pred.cpu().numpy()

            # Error
            diff = u_true - u_pred
            err_norm = np.linalg.norm(diff, axis=1)
            true_norm = np.linalg.norm(u_true, axis=1)

            mask_moving = true_norm > 1e-3
            rel_error = np.zeros_like(err_norm)
            rel_error[mask_moving] = (
                err_norm[mask_moving] / true_norm[mask_moving]
            ) * 100.0

            # Physics parameters
            E_val = data.x[0, 2].item() * float(norm["E"]) / 1e9
            nu_val = data.x[0, 3].item() * float(norm["nu"])
            force_x = data.x[:, 4].abs().max().item() * float(norm["force"])

            current_id = i
            if hasattr(data, "sim_id"):
                sid = data.sim_id
                if isinstance(sid, torch.Tensor):
                    current_id = int(sid.item())
                elif isinstance(sid, (list, np.ndarray)):
                    current_id = int(sid[0])
                else:
                    current_id = int(sid)

            sim_stats.append(
                {
                    "ID": current_id,
                    "MAE (mm)": np.mean(err_norm),
                    "MSE": np.mean(err_norm**2),
                    "Max Error (mm)": np.max(err_norm),
                    "Rel Error (%)": (
                        np.mean(rel_error[mask_moving]) if np.any(mask_moving) else 0.0
                    ),
                    "Young (GPa)": E_val,
                    "Poisson": nu_val,
                    "Force (N)": force_x,
                    "Nodes": data.num_nodes,
                }
            )

            # Spatial
            coords = data.x[:, 0:2].cpu().numpy()
            coords[:, 0] *= float(norm["x"])
            coords[:, 1] *= float(norm["y"])
            all_x.append(coords[:, 0])
            all_y.append(coords[:, 1])
            all_error_abs.append(err_norm)
            all_error_rel.append(rel_error)

            count += 1
            progress_bar.progress(count / num_samples)

    progress_bar.empty()

    spatial_data = {
        "x": np.concatenate(all_x),
        "y": np.concatenate(all_y),
        "err_abs": np.concatenate(all_error_abs),
        "err_rel": np.concatenate(all_error_rel),
    }

    return pd.DataFrame(sim_stats), spatial_data


def plot_spatial_heatmap(spatial_data, metric="err_abs", bins=(100, 45)):
    """
    Generate a heatmap
    """
    x, y, z = spatial_data["x"], spatial_data["y"], spatial_data[metric]

    # Calculate statistic on fixed grid
    stat = binned_statistic_2d(
        x, y, z, statistic="mean", bins=bins, range=[[0, 2.5], [-0.6, 0.6]]
    )

    # Filling the hole with white/transparent
    heatmap_data = stat.statistic.T

    masked_data = np.ma.masked_invalid(heatmap_data)

    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.cm.get_cmap("hot_r").copy()
    cmap.set_bad(color="white", alpha=0)

    im = ax.imshow(
        masked_data,
        origin="lower",
        extent=[0, 2.5, -0.6, 0.6],
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, ax=ax)
    label = "Absolute Error (mm)" if metric == "err_abs" else "Relative Error (%)"
    cbar.set_label(label)

    ax.set_title(f"Spatial Distribution of the Error")
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")

    return fig


def plot_individual_sample(model, dataset, device, sample_idx, scale_factor, norm_cfg):
    data = dataset[sample_idx]
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        pred = model(batch).cpu().numpy()
    u_true = data.y.numpy()
    pos = data.x[:, 0:2].numpy()
    pos[:, 0] *= float(norm_cfg["x"])
    pos[:, 1] *= float(norm_cfg["y"])
    error = np.linalg.norm(u_true - pred, axis=1)
    triang = tri.Triangulation(pos[:, 0], pos[:, 1])
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].triplot(triang, "k-", alpha=0.1)
    pos_t = pos + (u_true / 1000.0) * scale_factor
    pos_p = pos + (pred / 1000.0) * scale_factor
    axs[0].triplot(
        pos_t[:, 0], pos_t[:, 1], triang.triangles, "b-", alpha=0.5, label="Ref"
    )
    axs[0].triplot(
        pos_p[:, 0], pos_p[:, 1], triang.triangles, "r--", alpha=0.8, label="GNN"
    )
    axs[0].legend()
    axs[1].tripcolor(triang, error, cmap="inferno", shading="gouraud")
    return fig


# --- APP ---
def main():
    st.title("Dashboard - GNN Analysis")
    dataset, model, cfg, device, status_msg, status_type = load_resources()

    if model is None:
        st.error(status_msg)
        return

    if status_type == "warning":
        st.warning(status_msg)
    else:
        st.sidebar.success(status_msg)

    n_anal = st.sidebar.slider("Simulations to analyze", 10, 500, 50)

    # Appel de la fonction v3 pour forcer la mise à jour
    df_raw, spatial_data = run_batch_analysis_v3(model, dataset, cfg, device, n_anal)

    # --- FILTRAGE DES ABERRATIONS ---
    st.sidebar.markdown("### Filters")
    use_filter = st.sidebar.checkbox("Exclude Divergences", value=True)
    if use_filter:
        threshold = st.sidebar.number_input(
            "Threshold of Max Relative Error (%)", value=100.0
        )
        df_stats = df_raw[df_raw["Rel Error (%)"] < threshold]
        n_excluded = len(df_raw) - len(df_stats)
        if n_excluded > 0:
            st.sidebar.warning(f"{n_excluded} simulation(s) exclue(s) (>{threshold}%)")
    else:
        df_stats = df_raw

    tab1, tab2, tab3, tab4 = st.tabs(["Metrics", "Space", "Training", "Detail"])

    with tab1:
        st.header("Global Preformance on filtered data")
        cols = st.columns(4)
        cols[0].metric("MAE Median", f"{df_stats['MAE (mm)'].median():.4f} mm")
        cols[1].metric("MSE Median", f"{df_stats['MSE'].median():.2e}")
        cols[2].metric(
            "Err Max (Worst Case)", f"{df_stats['Max Error (mm)'].max():.4f} mm"
        )
        cols[3].metric("Err Rel Median", f"{df_stats['Rel Error (%)'].median():.2f} %")

        st.divider()
        cx = st.selectbox(
            "Parameter X", ["Force (N)", "Young (GPa)", "Poisson", "Nodes"]
        )
        cy = st.selectbox("Metric Y", ["MAE (mm)", "Rel Error (%)", "Max Error (mm)"])

        # Sécurité pour Plotly : vérifier que ID est bien là
        h_data = ["ID"] if "ID" in df_stats.columns else None

        fig = px.scatter(
            df_stats,
            x=cx,
            y=cy,
            color="Young (GPa)",
            size="Nodes",
            hover_data=h_data,
            title=f"Correlation : {cy} vs {cx}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        type_err = st.radio("Type", ["Absolute (mm)", "Relative (%)"], horizontal=True)
        metric_key = "err_abs" if "Absolute" in type_err else "err_rel"
        st.pyplot(plot_spatial_heatmap(spatial_data, metric=metric_key))

    with tab3:
        hist = load_training_history(cfg)
        if hist is not None:
            fig = px.line(
                hist, x="epoch", y=["total_loss", "data_loss", "phys_loss"], log_y=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No History")

    with tab4:
        col_sel, col_view = st.columns([1, 3])
        with col_sel:
            # Allows to chose among disponible IDs
            sid = st.number_input("Index Simulation", 0, len(dataset) - 1, 0)
            amp = st.slider("Amplification", 1.0, 2000.0, 500.0)

            data_s = dataset[sid]
            E_disp = data_s.x[0, 2] * float(cfg["normalization"]["E"]) / 1e9
            st.info(f"Young: {E_disp:.1f} GPa")

        with col_view:
            st.pyplot(
                plot_individual_sample(
                    model, dataset, device, sid, amp, cfg["normalization"]
                )
            )


if __name__ == "__main__":
    main()
