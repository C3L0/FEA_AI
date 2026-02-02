import os
import signal
import sys

import gmsh
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import streamlit as st
import torch

# --- FIX IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from torch_geometric.data import Batch, Data

from src.fea_gnn.model import HybridPhysicsGNN
from src.fea_gnn.utils import load_config

# --- PAGE CONFIG ---
st.set_page_config(page_title="FEA-GNN: Instant Simulator", layout="wide")

# --- MATERIALS DICTIONARY ---
MATERIALS = {
    "Custom": {"E": 210.0, "nu": 0.3},
    "Steel": {"E": 210.0, "nu": 0.30},
    "Aluminum": {"E": 69.0, "nu": 0.33},
    "Titanium": {"E": 110.0, "nu": 0.34},
    "Concrete": {"E": 30.0, "nu": 0.20},
    "Wood (Oak)": {"E": 12.0, "nu": 0.30},
    "Rubber": {"E": 0.1, "nu": 0.49},  # Warning: Out of training distribution
}

# --- UTILITY FUNCTIONS ---


@st.cache_resource
def load_model():
    """Load the trained GNN model."""
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridPhysicsGNN(
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["layers"],
        input_dim=cfg["model"]["input_dim"],
    ).to(device)

    path = os.path.join(cfg["env"]["save_path"], "gnn_hybrid_scaled.pth")

    if not os.path.exists(path):
        return None, cfg, device, "Model not found"

    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    except RuntimeError:
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=False
        )

    model.eval()
    return model, cfg, device, "OK"


def generate_mesh_live(L, H, R, res_factor=10):
    """
    Generate a GMSH mesh on the fly without using FEniCS.
    Includes a workaround for the 'signal only works in main thread' error.
    Returns: coords (N,2), cells (M,3)
    """
    # WORKAROUND: Streamlit runs in a thread, so we bypass GMSH's signal handler registration
    original_handler = signal.signal
    signal.signal = lambda *args, **kwargs: None

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add("Live_Mesh")

        lc = H / res_factor

        # Geometry
        rect = model.occ.addRectangle(0, 0, 0, L, H)
        circle = model.occ.addDisk(L / 2, H / 2, 0, R, R)
        out, _ = model.occ.cut([(2, rect)], [(2, circle)])
        model.occ.synchronize()

        # Mesh generation
        model.mesh.setSize(model.getEntities(0), lc)
        model.mesh.generate(2)

        # Extract Nodes
        node_tags, node_coords, _ = model.mesh.getNodes()
        nodes = np.array(node_coords).reshape(-1, 3)[:, :2]

        # Extract Elements (Triangles)
        element_types, element_tags, node_tags_per_element = model.mesh.getElements(
            dim=2
        )

        if len(element_types) == 0:
            return None, None

        # Look for triangular elements (Type 2)
        tris = None
        for i, t in enumerate(element_types):
            if t == 2:
                tris = node_tags_per_element[i].reshape(-1, 3)
                break

        if tris is None:
            return None, None

        # Map GMSH tags (1-based) to 0-based indices
        tag_map = {tag: i for i, tag in enumerate(node_tags)}
        mapped_tris = np.array([[tag_map[t] for t in tri_row] for tri_row in tris])

        return nodes, mapped_tris

    finally:
        # Finalize and restore the original signal handler
        if gmsh.isInitialized():
            gmsh.finalize()
        signal.signal = original_handler


def prepare_graph(nodes, cells, E, nu, Fx, Fy, cfg):
    """Convert raw mesh into normalized PyG Data object."""
    norm = cfg["normalization"]
    num_nodes = len(nodes)

    # 1. Features: [x, y, E, nu, Fx, Fy, isFixed]
    features = np.zeros((num_nodes, 7))
    features[:, 0] = nodes[:, 0]
    features[:, 1] = nodes[:, 1]
    features[:, 2] = E * 1e9  # GPa to Pa
    features[:, 3] = nu

    # Boundary Conditions: Fixed on left (x ~ 0)
    is_fixed = np.isclose(nodes[:, 0], 0.0, atol=1e-2)
    features[:, 6] = is_fixed.astype(float)

    # Force on right (x ~ L)
    max_x = np.max(nodes[:, 0])
    is_loaded = np.isclose(nodes[:, 0], max_x, atol=1e-2)
    features[is_loaded, 4] = Fx
    features[is_loaded, 5] = Fy

    # Normalize inputs
    pos_physical = torch.tensor(features[:, 0:2], dtype=torch.float)
    feat_norm = features.copy()
    feat_norm[:, 0] /= float(norm["x"])
    feat_norm[:, 1] /= float(norm["y"])
    feat_norm[:, 2] /= float(norm["E"])
    feat_norm[:, 3] /= float(norm["nu"])
    feat_norm[:, 4] /= float(norm["force"])
    feat_norm[:, 5] /= float(norm["force"])

    x = torch.tensor(feat_norm, dtype=torch.float)

    # 2. Edges
    edges = []
    for tri_cell in cells:
        n1, n2, n3 = tri_cell
        edges.extend([[n1, n2], [n2, n1], [n2, n3], [n3, n2], [n3, n1], [n1, n3]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Edge Attributes (Physical distances)
    row, col = edge_index
    edge_vector = pos_physical[col] - pos_physical[row]
    edge_len = torch.norm(edge_vector, dim=1, keepdim=True)
    edge_attr = torch.cat([edge_vector, edge_len], dim=1)

    batch = torch.zeros(num_nodes, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


# --- INTERFACE ---


def main():
    st.sidebar.title("Parameters")

    model, cfg, device, status = load_model()
    if model is None:
        st.error(f"Initialization Error: {status}")
        return

    # 1. User Inputs
    st.sidebar.subheader("1. Geometry")
    L = st.sidebar.slider("Length L (m)", 1.0, 3.0, 2.0, 0.1)
    H = st.sidebar.slider("Height H (m)", 0.3, 1.0, 0.5, 0.05)
    max_R = min(L, H) / 2.0 * 0.8
    R = st.sidebar.slider("Hole Radius R (m)", 0.05, max_R, 0.1, 0.01)

    # Mesh Resolution Slider
    st.sidebar.subheader("2. Mesh Resolution")
    res_factor = st.sidebar.slider(
        "Density Factor",
        5,
        30,
        15,
        help="Higher value means more nodes and smaller triangles.",
    )

    st.sidebar.subheader("3. Material")
    mat_name = st.sidebar.selectbox("Preset", list(MATERIALS.keys()), index=1)

    defaults = MATERIALS[mat_name]
    E_gpa = st.sidebar.number_input("Young Modulus (GPa)", 0.1, 300.0, defaults["E"])
    nu = st.sidebar.number_input("Poisson Ratio", 0.0, 0.49, defaults["nu"])

    if E_gpa < 10.0:
        st.sidebar.warning(
            "Warning: E < 10 GPa is outside training range. Results may be unreliable."
        )

    st.sidebar.subheader("4. Loading")
    F_val = st.sidebar.slider("Tensile Force (N)", 1000.0, 5e6, 1e6, step=10000.0)

    # Visual Amplification setting
    st.sidebar.subheader("5. Visualization")
    scale_factor = st.sidebar.slider(
        "Amplification Factor",
        1.0,
        100.0,
        1.0,
        help="Magnifies the displacement for easier viewing.",
    )

    # 6. Action Button
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Meshing & AI Prediction..."):
            # Generate mesh using user-selected resolution
            nodes, cells = generate_mesh_live(L, H, R, res_factor=res_factor)

            if nodes is None:
                st.error("GMSH meshing failed.")
                return

            # Graph preparation
            data = prepare_graph(nodes, cells, E_gpa, nu, F_val, 0.0, cfg)

            # Inference
            data = data.to(device)
            with torch.no_grad():
                pred_raw = model(data).cpu().numpy()

            # Conversion to mm
            u_mm = pred_raw
            mag_mm = np.linalg.norm(u_mm, axis=1)

            # Visualisation
            st.success(f"Calculation finished in milliseconds ({len(nodes)} nodes)")

            col1, col2 = st.columns([3, 1])

            with col1:
                # Create figure with two subplots: Initial and Deformed
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

                # Plot 1: Initial Structure (Undeformed)
                triang_init = tri.Triangulation(
                    nodes[:, 0], nodes[:, 1], triangles=cells
                )
                ax1.triplot(triang_init, "b-", linewidth=0.5, alpha=0.4)
                ax1.set_aspect("equal")
                ax1.set_title("Initial Undeformed Mesh", fontsize=12, fontweight="bold")
                ax1.axis("off")

                # Plot 2: Predicted Structure (Deformed)
                # Deformed coordinates
                pos_def = nodes + (u_mm / 1000.0) * scale_factor
                triang_def = tri.Triangulation(
                    pos_def[:, 0], pos_def[:, 1], triangles=cells
                )

                tpc = ax2.tripcolor(triang_def, mag_mm, shading="gouraud", cmap="jet")
                ax2.triplot(triang_def, "k-", linewidth=0.1, alpha=0.3)

                ax2.set_aspect("equal")
                ax2.set_title(
                    f"GNN Result: {mat_name} Plate (Deformed x{scale_factor})",
                    fontsize=12,
                    fontweight="bold",
                )
                ax2.axis("off")

                # Colorbar for the deformed plot
                plt.colorbar(
                    tpc,
                    ax=ax2,
                    label="Total Displacement (mm)",
                    orientation="horizontal",
                    pad=0.08,
                )

                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.markdown("### Results")
                st.metric("Max Displacement", f"{np.max(mag_mm):.4f} mm")
                st.metric("Mean Displacement", f"{np.mean(mag_mm):.4f} mm")
                st.markdown("---")
                st.markdown("**Mesh Info:**")
                st.text(f"Nodes: {len(nodes)}")
                st.text(f"Elements: {len(cells)}")

    else:
        st.info("Configure your plate and click 'Run Simulation'")
        st.markdown(
            """
        ### Welcome to the AI Simulator
        This demonstrator uses a Graph Neural Network (GNN) to predict structural deformation instantly.
        
        **How it works:**
        1. GMSH generates a unique mesh based on your parameters.
        2. The GNN propagates forces through the nodes.
        3. Results are displayed in real-time (< 0.1s).
        """
        )


if __name__ == "__main__":
    main()
