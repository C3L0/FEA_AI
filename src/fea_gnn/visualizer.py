import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from src.fea_gnn.utils import load_config


def plot_fea_comparison(model, dataset, device, sample_idx=0, amp_factor=None):
    """
    Visualisation corrigée avec conversion d'unités propre.
    """
    cfg = load_config()
    norm = cfg["normalization"]

    model.eval()
    data = dataset[sample_idx].to(device)

    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        raw_pred = model(data).cpu().numpy()

    # --- DÉ-NORMALISATION GÉOMÉTRIE ---
    coords_m = data.x[:, 0:2].cpu().numpy()
    coords_m[:, 0] *= float(norm["x"])
    coords_m[:, 1] *= float(norm["y"])

    # Sorties u, v (déjà en mm via target_scale=1000 dans le pipeline)
    gt_disp_mm = data.y.cpu().numpy()
    pred_disp_mm = raw_pred

    # --- CORRECTION GPa ---
    # data.x[:, 2] est ~1.0. norm['E'] est 210e9.
    # (1.0 * 210e9) / 1e9 = 210 GPa.
    E_val_gpa = (data.x[:, 2].mean().item() * float(norm["E"])) / 1e9

    # --- AMPLIFICATION ---
    x_coords, y_coords = coords_m[:, 0], coords_m[:, 1]
    if amp_factor is None:
        # On se base sur FEniCS pour l'amplification stable
        max_gt_m = np.max(np.linalg.norm(gt_disp_mm, axis=1)) / 1000.0
        H = np.max(y_coords) - np.min(y_coords)
        amp_factor = (H * 0.15) / max_gt_m if max_gt_m > 1e-9 else 1.0

    triang = mtri.Triangulation(x_coords, y_coords)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=120)

    def draw_structure(ax, disp_mm, title, color_map, vmin, vmax):
        disp_m = disp_mm / 1000.0
        x_def = x_coords + disp_m[:, 0] * amp_factor
        y_def = y_coords + disp_m[:, 1] * amp_factor
        mag_mm = np.linalg.norm(disp_mm, axis=1)

        tpc = ax.tripcolor(
            x_def,
            y_def,
            triang.triangles,
            mag_mm,
            shading="flat",
            cmap=color_map,
            vmin=vmin,
            vmax=vmax,
        )
        ax.triplot(x_def, y_def, triang.triangles, "k-", linewidth=0.1, alpha=0.15)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")
        return tpc

    # On utilise la même échelle de couleur pour comparer honnêtement
    vmax = np.max(gt_disp_mm)
    vmin = 0

    im1 = draw_structure(
        ax1, gt_disp_mm, "VÉRITÉ TERRAIN (FEniCS)", "Blues", vmin, vmax
    )
    plt.colorbar(im1, ax=ax1, label="Déplacement (mm)", fraction=0.02)

    error_mm = np.mean(np.linalg.norm(gt_disp_mm - pred_disp_mm, axis=1))
    im2 = draw_structure(
        ax2,
        pred_disp_mm,
        f"PRÉDICTION GNN (Erreur moy: {error_mm:.4f} mm)",
        "Reds",
        vmin,
        vmax,
    )
    plt.colorbar(im2, ax=ax2, label="Déplacement (mm)", fraction=0.02)

    plt.suptitle(
        f"Analyse Finale | Matériau: {E_val_gpa:.1f} GPa\n(Déformations amplifiées x{amp_factor:.1f})",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
