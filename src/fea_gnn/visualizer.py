import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch


def plot_fea_comparison(model, dataset, device, sample_idx=0, amp_factor=None):
    """
    Visualisation pour maillages non-structurés (Triangles).
    Compatible avec les objets Data de PyTorch Geometric.
    """
    model.eval()

    # 1. Récupération de l'objet Data
    # dataset[i] retourne un objet Data(x=..., edge_index=..., y=...)
    data = dataset[sample_idx]

    # Pour le modèle, il faut définir un vecteur 'batch' (tous les nœuds appartiennent au graphe 0)
    data = data.to(device)
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        # Prédiction brute (x1000)
        raw_pred = model(data).cpu().numpy()

    # 2. Mise à l'échelle (Retour en mètres)
    SCALE = 1000.0
    pred_disp_m = raw_pred / SCALE
    gt_disp_m = data.y.cpu().numpy() / SCALE

    # 3. Extraction de la géométrie (x, y)
    # data.x contient [x, y, E, nu, Fx, Fy, isFixed]
    coords = data.x[:, 0:2].cpu().numpy()
    x_coords, y_coords = coords[:, 0], coords[:, 1]

    # Propriétés matérielles pour le titre (E est index 3, nu index 4)
    # On prend la moyenne car c'est constant par simulation
    E_val = data.x[:, 3].mean().item() * 210.0
    nu_val = data.x[:, 4].mean().item() * 0.5

    # 4. Calcul de l'amplification automatique
    if amp_factor is None:
        max_disp = np.max(np.linalg.norm(gt_disp_m, axis=1))
        L_approx = np.max(x_coords)  # Longueur approximative
        if max_disp < 1e-9:
            max_disp = 1.0
        amp_factor = (L_approx * 0.15) / max_disp

    print(
        f"[Visualizer] Sample {sample_idx} | E={E_val:.1f} GPa | Amp: x{amp_factor:.1f}"
    )

    # 5. Création de la triangulation pour l'affichage
    # On utilise une triangulation de Delaunay basée sur les coordonnées xy
    triang = mtri.Triangulation(x_coords, y_coords)

    # Fonction de dessin interne
    def plot_mesh(ax, disp, title, cmap):
        # Coordonnées déformées
        x_def = x_coords + disp[:, 0] * amp_factor
        y_def = y_coords + disp[:, 1] * amp_factor

        magnitude = np.linalg.norm(disp, axis=1)

        # On dessine les triangles remplis (tripcolor)
        trip = ax.tripcolor(
            x_def, y_def, triang.triangles, magnitude, shading="gouraud", cmap=cmap
        )

        # On ajoute le maillage fil de fer par-dessus pour voir la structure
        ax.triplot(x_def, y_def, triang.triangles, "k-", linewidth=0.1, alpha=0.3)

        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.axis("off")  # On cache les axes pour un look plus propre
        return trip

    # 6. Affichage
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Vérité Terrain
    im1 = plot_mesh(ax1, gt_disp_m, "Vérité Terrain (FEniCS)", "Blues")
    plt.colorbar(im1, ax=ax1, label="Déplacement (m)")

    # Prédiction
    im2 = plot_mesh(
        ax2,
        pred_disp_m,
        f"Prédiction GNN (Erreur Moy: {np.mean(np.abs(gt_disp_m - pred_disp_m)):.2e} m)",
        "Reds",
    )
    plt.colorbar(im2, ax=ax2, label="Déplacement (m)")

    plt.suptitle(f"Comparaison Plaque Trouée (E={E_val:.0f} GPa)", fontsize=16)
    plt.tight_layout()

    # Sauvegarde automatique
    plt.savefig(f"visualization/eval_sample_{sample_idx}.png")
    plt.show()


# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
#
#
# def plot_fea_comparison(model, dataset, device, sample_idx=0, amp_factor=None):
#     """
#     Generates a professional side-by-side comparison of Ground Truth vs Prediction.
#     Uses 'Quad' elements for a realistic FEA look.
#     """
#     model.eval()
#
#     # 1. Prepare Data
#     test_x, test_y_gt = dataset[sample_idx]
#
#     with torch.no_grad():
#         # Add batch dim, move to GPU, predict, remove batch dim, move to CPU
#         raw_pred = model(test_x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
#
#     # 2. Handle Scaling (Crucial: Convert back to Meters)
#     # The model predicts x1000 values, we need real physics units for plotting
#     SCALE = 1000.0
#     pred_disp_m = raw_pred / SCALE
#     gt_disp_m = test_y_gt.numpy() / SCALE
#
#     # 3. Geometry & Material Info
#     coords = dataset.coords
#     nx, ny = dataset.nx, dataset.ny
#
#     # Extract material properties for title (E is at index 3, nu at index 4)
#     E_val = test_x[0, 3].item() * 210.0  # Un-normalize (assuming max 210 GPa)
#     nu_val = test_x[0, 4].item() * 0.5  # Un-normalize (assuming max 0.5)
#
#     # 4. Auto-Amplification Calculation
#     if amp_factor is None:
#         max_disp = np.max(np.linalg.norm(gt_disp_m, axis=1))
#         # Target: Max displacement should look like ~15% of beam length
#         if max_disp < 1e-9:
#             max_disp = 1.0  # Avoid div/0
#         amp_factor = (dataset.length * 0.15) / max_disp
#
#     print(
#         f"[Visualizer] Sample {sample_idx} | E={E_val:.1f} GPa | Amplification: x{amp_factor:.1f}"
#     )
#
#     # 5. Plotting Logic
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
#
#     def _draw_mesh_on_ax(ax, displacements, title, cmap_name):
#         # Calculate deformed positions
#         deformed_pos = coords + displacements * amp_factor
#         magnitude = np.linalg.norm(displacements, axis=1)
#
#         # Fixed Plot Limits (to keep scale consistent)
#         ax.set_xlim(-0.2, dataset.length * 1.2)
#         ax.set_ylim(-dataset.height * 2.0, dataset.height * 2.0)
#
#         # Draw Quads (Elements)
#         for i in range(ny - 1):
#             for j in range(nx - 1):
#                 # Map grid indices to linear indices
#                 # (i, j) -> top-left of the quad
#                 n1 = i * nx + j
#                 n2 = i * nx + (j + 1)
#                 n3 = (i + 1) * nx + (j + 1)
#                 n4 = (i + 1) * nx + j
#
#                 quad_points = [
#                     deformed_pos[n1],
#                     deformed_pos[n2],
#                     deformed_pos[n3],
#                     deformed_pos[n4],
#                 ]
#
#                 # Color based on average displacement of the 4 nodes
#                 avg_mag = np.mean(
#                     [magnitude[n1], magnitude[n2], magnitude[n3], magnitude[n4]]
#                 )
#                 norm_mag = avg_mag / (np.max(magnitude) + 1e-9)
#
#                 poly = patches.Polygon(
#                     quad_points,
#                     closed=True,
#                     linewidth=0.5,
#                     edgecolor="black",
#                     facecolor=plt.cm.get_cmap(cmap_name)(norm_mag),
#                     alpha=0.85,
#                 )
#                 ax.add_patch(poly)
#
#         ax.set_title(title, fontsize=12, fontweight="bold")
#         ax.set_aspect("equal")
#
#         # Colorbar
#         sm = plt.cm.ScalarMappable(
#             cmap=cmap_name, norm=plt.Normalize(vmin=0, vmax=np.max(magnitude))
#         )
#         plt.colorbar(sm, ax=ax, label="Displacement (m)", shrink=0.8)
#
#     # Draw both
#     _draw_mesh_on_ax(ax1, gt_disp_m, f"Ground Truth (FEA)", "Blues")
#     _draw_mesh_on_ax(ax2, pred_disp_m, f"GNN Prediction", "Reds")
#
#     plt.suptitle(f"Material: E={E_val:.1f} GPa, ν={nu_val:.2f}", fontsize=14)
#     plt.tight_layout()
#     plt.show()
