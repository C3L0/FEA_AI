import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
from train import PhysicsAwareTransformer

# Constantes de dénormalisation
LENGTH = 2.0
HEIGHT = 0.2
MAX_DISP_SCALE = 0.1
MAX_STRESS_SCALE = 50e6

# Propriétés du matériau pour la rupture
YIELD_STRENGTH_WOOD = 40e6 # Pa (40 MPa limite élastique approx du pin)

def visualize_prediction():
    # Détection du processeur (Optimisé pour RTX 5070 Ti)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exécution sur : {device}")
    
    # 1. Charger le modèle
    model = PhysicsAwareTransformer().to(device)
    try:
        model.load_state_dict(torch.load('mesh_transformer.pth', map_location=device))
    except FileNotFoundError:
        print("Erreur : Modèle 'mesh_transformer.pth' introuvable. Veuillez d'abord entraîner le modèle.")
        return
    model.eval()

    # 2. Créer un scénario de test unique
    NODES_X, NODES_Y = 20, 6
    xs = np.linspace(0, LENGTH, NODES_X)
    ys = np.linspace(0, HEIGHT, NODES_Y)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    mesh_x_flat = mesh_x.flatten()
    mesh_y_flat = mesh_y.flatten()
    
    # --- PARAMÈTRES DE TEST ---
    TEST_FORCE_POS = 1.8  # Position de la force (m)
    TEST_FORCE_MAG = 8500 # Magnitude de la force (N)
    # --------------------------

    # Préparation des données d'entrée (Normalisation)
    norm_x = mesh_x_flat / LENGTH
    norm_y = mesh_y_flat / HEIGHT
    norm_f_pos = TEST_FORCE_POS / LENGTH
    norm_f_mag = (TEST_FORCE_MAG - 1000) / 9000
    
    input_tensor = np.stack([
        norm_x, norm_y, 
        np.full_like(norm_x, norm_f_pos), 
        np.full_like(norm_x, norm_f_mag)
    ], axis=1)
    
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)

    # 3. Prédiction (Inférence)
    with torch.no_grad():
        # Synchronisation CUDA pour une mesure précise du temps sur GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        prediction = model(input_tensor) # [1, Num_Nodes, 2]
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()

    # Calcul du temps en millisecondes
    duration_ms = (end_time - start_time) * 1000
    print(f"Temps d'inférence : {duration_ms:.2f} ms")

    # Récupérer les résultats et dénormaliser
    pred_numpy = prediction.cpu().squeeze(0).numpy()
    pred_disp_y = pred_numpy[:, 0] * MAX_DISP_SCALE
    pred_stress = pred_numpy[:, 1] * MAX_STRESS_SCALE

    # Calcul des coordonnées déformées pour la visualisation (facteur d'exagération)
    scale_factor = 5.0 
    deformed_x = mesh_x_flat
    deformed_y = mesh_y_flat + (pred_disp_y * scale_factor)

    # 4. Vérification de la rupture
    max_stress = np.max(np.abs(pred_stress))
    rupture = max_stress > YIELD_STRENGTH_WOOD
    status_color = 'red' if rupture else 'green'
    status_text = "RUPTURE DÉTECTÉE !" if rupture else "Intégrité Structurelle OK"

    # 5. Affichage avec Matplotlib
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Triangulation pour le rendu de la surface
    triang = tri.Triangulation(deformed_x, deformed_y)

    # Graphique 1 : Carte de chaleur des contraintes (Stress Map)
    cmap = plt.get_cmap('jet')
    tripcolor = ax1.tripcolor(triang, pred_stress/1e6, cmap=cmap, shading='gouraud') # MPa
    ax1.scatter(TEST_FORCE_POS, HEIGHT + (pred_disp_y.min() * scale_factor), color='black', marker='v', s=100, label='Point d\'application')
    ax1.set_title(f"Simulation Mesh Transformer : Contraintes & Déformée (Echelle x{scale_factor})")
    ax1.set_xlabel("Longueur (m)")
    ax1.set_ylabel("Hauteur (m)")
    cbar = plt.colorbar(tripcolor, ax=ax1)
    cbar.set_label('Contrainte de flexion (MPa)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Graphique 2 : Profil de la fibre neutre
    mid_idx = np.isclose(mesh_y_flat, HEIGHT/2, atol=0.01)
    # Trier par X pour un tracé propre
    sort_idx = np.argsort(mesh_x_flat[mid_idx])
    # Correction : Utilisation de 'alpha' au lieu de 'markers_alpha'
    ax2.plot(mesh_x_flat[mid_idx][sort_idx], pred_disp_y[mid_idx][sort_idx]*1000, 'b-o', alpha=0.6, label='Déplacement (mm)')
    ax2.set_title("Profil de déflexion centrale")
    ax2.set_ylabel("Déplacement vertical (mm)")
    ax2.set_xlabel("Position sur la planche (m)")
    ax2.grid(True)
    ax2.legend()

    # Affichage du statut final
    plt.figtext(0.5, 0.02, 
                f"Scénario : Force de {TEST_FORCE_MAG} N à {TEST_FORCE_POS} m\n"
                f"Contrainte Max : {max_stress/1e6:.2f} MPa (Limite Bois : {YIELD_STRENGTH_WOOD/1e6} MPa)\n"
                f"Résultat : {status_text}", 
                ha="center", fontsize=12, fontweight='bold',
                bbox={"facecolor": status_color, "alpha": 0.2, "pad": 10})

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()

if __name__ == "__main__":
    visualize_prediction()