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
