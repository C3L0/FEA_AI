import torch
import numpy as np
from torch_geometric.data import Data
import os

# --- Paramètres Physiques (Bois) ---
LENGTH = 1.0        # Longueur (m)
HEIGHT = 0.2        # Hauteur (m)
THICKNESS = 0.1     # Epaisseur (m)
E_MODULUS = 11e9    # Module de Young du bois (Pa) ~11 GPa
I_INERTIA = (THICKNESS * HEIGHT**3) / 12  # Moment d'inertie

# --- Paramètres du Maillage ---
NODES_X = 30        # Nombre de nœuds en X
NODES_Y = 4         # Nombre de nœuds en Y
NUM_SAMPLES = 1000  # Nombre de simulations à générer

def get_mesh_graph():
    """Crée la topologie du graphe (nœuds et arêtes) pour une grille régulière."""
    x = np.linspace(0, LENGTH, NODES_X)
    y = np.linspace(-HEIGHT/2, HEIGHT/2, NODES_Y)
    xx, yy = np.meshgrid(x, y)
    pos = np.vstack((xx.flatten(), yy.flatten())).T # Shape: [num_nodes, 2]
    
    # Création des arêtes (connectivité type grille)
    edges = []
    num_nodes = NODES_X * NODES_Y
    for i in range(num_nodes):
        # Connexion horizontale (droite)
        if (i + 1) % NODES_X != 0:
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        # Connexion verticale (bas)
        if i + NODES_X < num_nodes:
            edges.append([i, i + NODES_X])
            edges.append([i + NODES_X, i])
            
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.tensor(pos, dtype=torch.float), edge_index

def analytical_cantilever(pos, force_x, force_val):
    """
    Calcule la déformation (dy) et la contrainte (stress) théoriques 
    selon Euler-Bernoulli pour une poutre encastrée.
    """
    x = pos[:, 0]
    y = pos[:, 1]
    
    # Initialisation
    displacement_y = np.zeros_like(x)
    stress_sigma = np.zeros_like(x)
    
    # On considère la force appliquée à force_x (distance du mur)
    # Formule déflexion: v(x) = (F*x^2 * (3a - x)) / (6EI) pour 0 <= x <= a
    # a = point d'application de la force
    
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        
        # Déplacement (Simplification: on ignore le déplacement si x > point d'application pour l'exemple)
        if xi <= force_x:
            dy = (force_val * xi**2 * (3*force_x - xi)) / (6 * E_MODULUS * I_INERTIA)
        else:
            # Après le point de force, la pente est constante
            dy_at_a = (force_val * force_x**2 * (3*force_x - force_x)) / (6 * E_MODULUS * I_INERTIA)
            theta_at_a = (force_val * force_x**2) / (2 * E_MODULUS * I_INERTIA)
            dy = dy_at_a + theta_at_a * (xi - force_x)
            
        displacement_y[i] = -dy # Vers le bas
        
        # Contrainte de flexion: Sigma = -My / I
        # Moment M(x) = F * (x - a)
        moment = force_val * (xi - force_x) if xi < force_x else 0
        stress_sigma[i] = abs((moment * yi) / I_INERTIA) # Valeur absolue pour simplifier la visualisation

    return displacement_y, stress_sigma

def generate_dataset():
    print(f"Génération de {NUM_SAMPLES} simulations...")
    dataset = []
    pos_base, edge_index = get_mesh_graph()
    
    for _ in range(NUM_SAMPLES):
        # 1. Scénario aléatoire
        force_val = np.random.uniform(1000, 5000)  # Force entre 1kN et 5kN
        force_node_idx_x = np.random.randint(int(NODES_X*0.5), NODES_X) # Force appliquée sur la 2ème moitié
        force_pos_x = pos_base[force_node_idx_x * NODES_Y, 0].item() # Récupérer la coord X physique
        
        # 2. Calculer Ground Truth (Labels)
        dy, stress = analytical_cantilever(pos_base.numpy(), force_pos_x, force_val)
        
        # 3. Features d'entrée (X) pour chaque nœud
        # [pos_x, pos_y, force_appliquee_ici?, distance_au_mur]
        x_features = torch.zeros((pos_base.shape[0], 4))
        x_features[:, 0:2] = pos_base # Coordonnées géométriques
        
        # On marque les nœuds où la force est appliquée (toute la colonne verticale à force_pos_x)
        # Simplification: on applique la force sur les nœuds du haut à cette position X
        nodes_at_force_x = np.where(np.isclose(pos_base[:, 0], force_pos_x))[0]
        
        # On distribue la force sur ces nœuds dans les features
        for idx in nodes_at_force_x:
             x_features[idx, 2] = force_val 

        x_features[:, 3] = pos_base[:, 0] # Distance au mur (redondant mais aide le GNN)

        # 4. Cibles (Y)
        # [Déplacement Y, Contrainte]
        y_target = torch.stack([torch.tensor(dy), torch.tensor(stress)], dim=1).float()
        
        data = Data(x=x_features, edge_index=edge_index, y=y_target, pos=pos_base)
        dataset.append(data)
        
    print("Génération terminée.")
    torch.save(dataset, 'wood_fem_dataset.pt')

if __name__ == "__main__":
    generate_dataset()