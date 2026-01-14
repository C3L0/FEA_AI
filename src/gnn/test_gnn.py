import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data
from generate_data import analytical_cantilever, get_mesh_graph, NODES_X, NODES_Y
from train_gnn import WoodStressGNN
import os

# Limite élastique du bois (ex: Pin) pour la rupture ~ 40 MPa
RUPTURE_THRESHOLD_PA = 40e6 

def visualize_prediction():
    # 1. Charger le modèle
    model = WoodStressGNN(in_channels=4, out_channels=2)
    try:
        model.load_state_dict(torch.load('wood_gnn_model.pth'))
        model.eval()
    except FileNotFoundError:
        print("Modèle non trouvé. Lancez l'entraînement d'abord.")
        return

    # 2. Créer un scénario de test unique
    pos_base, edge_index = get_mesh_graph()
    force_val = 4500.0 # Force élevée pour tester la rupture
    force_pos_x = 1.8  # Bout de la planche
    
    # Préparer input pour le modèle
    x_features = torch.zeros((pos_base.shape[0], 4))
    x_features[:, 0:2] = pos_base
    nodes_at_force_x = np.where(np.isclose(pos_base[:, 0], force_pos_x, atol=0.1))[0] # Approx grid match
    
    if len(nodes_at_force_x) > 0:
        # On applique la force sur la grille la plus proche
        for idx in nodes_at_force_x:
            x_features[idx, 2] = force_val
            
    x_features[:, 3] = pos_base[:, 0]
    
    data = Data(x=x_features, edge_index=edge_index)
    
    # 3. Prédiction IA
    with torch.no_grad():
        pred = model(data) # [num_nodes, 2] -> (dy, stress)
    
    pred_disp = pred[:, 0].numpy()
    pred_stress = pred[:, 1].numpy()
    
    # 4. Vérité Terrain (Physique) pour comparaison
    true_disp, true_stress = analytical_cantilever(pos_base.numpy(), force_pos_x, force_val)
    
    # --- Analyse de Rupture ---
    max_stress_pred = np.max(pred_stress)
    rupture_detected = max_stress_pred > RUPTURE_THRESHOLD_PA
    status_color = 'red' if rupture_detected else 'green'
    status_text = "RUPTURE DÉTECTÉE !" if rupture_detected else "Structure Intègre"
    
    print(f"Force appliquée: {force_val} N à x={force_pos_x}m")
    print(f"Contrainte Max Prédite: {max_stress_pred/1e6:.2f} MPa")
    print(f"Seuil de rupture: {RUPTURE_THRESHOLD_PA/1e6:.2f} MPa")
    print(f"Statut: {status_text}")

    # --- Visualisation ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    
    # Coordonnées déformées (facteur d'échelle exagéré pour voir la déformation)
    scale_factor = 50 
    x_def = pos_base[:, 0].numpy()
    y_def_true = pos_base[:, 1].numpy() + true_disp * scale_factor
    y_def_pred = pos_base[:, 1].numpy() + pred_disp * scale_factor

    # 1. Déplacement Réel
    sc1 = axes[0, 0].scatter(x_def, y_def_true, c=true_disp, cmap='viridis', s=20)
    axes[0, 0].set_title("Vraie Déformation (Physique)")
    plt.colorbar(sc1, ax=axes[0, 0], label='Déplacement Y (m)')
    
    # 2. Déplacement Predit
    sc2 = axes[0, 1].scatter(x_def, y_def_pred, c=pred_disp, cmap='viridis', s=20)
    axes[0, 1].set_title("Déformation Prédite (GNN)")
    plt.colorbar(sc2, ax=axes[0, 1], label='Déplacement Y (m)')
    
    # 3. Contrainte Réelle
    sc3 = axes[1, 0].scatter(x_def, y_def_true, c=true_stress, cmap='inferno', s=20)
    axes[1, 0].set_title("Vraie Contrainte (Von Mises)")
    plt.colorbar(sc3, ax=axes[1, 0], label='Contrainte (Pa)')

    # 4. Contrainte Prédite
    sc4 = axes[1, 1].scatter(x_def, y_def_pred, c=pred_stress, cmap='inferno', s=20)
    axes[1, 1].set_title(f"Contrainte Prédite (GNN)\n{status_text}")
    plt.colorbar(sc4, ax=axes[1, 1], label='Contrainte (Pa)')
    
    # Ajout du texte de rupture sur le graphe
    axes[1, 1].text(0.5, 0.5, status_text, 
                    transform=axes[1, 1].transAxes, 
                    color=status_color, 
                    fontsize=12, weight='bold', ha='center')

    plt.tight_layout()
    #plt.show()
    plt.savefig("../../visualization/gnn_prediction.png")
    '''
    # Sauvegarde de la figure dans le dossier 'visualization' du dossier parent
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(parent_dir, '..', 'visualization')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "gnn_prediction.png"))
    '''


if __name__ == "__main__":
    visualize_prediction()