import torch
import numpy as np
import os

# Configuration Physique (Poutre en Bois)
LENGTH = 2.0        # Longueur (m)
HEIGHT = 0.2        # Hauteur (m)
THICKNESS = 0.1     # Epaisseur (m)
E_MODULUS = 12e9    # Module de Young (Bois ~12 GPa)
I_INERTIA = (THICKNESS * HEIGHT**3) / 12 # Moment quadratique

# Paramètres du Dataset
NUM_SAMPLES = 2000  # Nombre de simulations
NODES_X = 20        # Résolution du maillage en X
NODES_Y = 6         # Résolution du maillage en Y
NUM_NODES = NODES_X * NODES_Y

def get_analytical_solution(force_x_pos, force_magnitude, mesh_x, mesh_y):
    """
    Calcule la déformation (v) et la contrainte (sigma) selon Euler-Bernoulli
    pour une poutre encastrée à x=0.
    """
    # Initialisation
    displacement_y = np.zeros_like(mesh_x)
    stress_sigma = np.zeros_like(mesh_x)
    
    # 1. Calcul du Déplacement (Flèche)
    # Formule poutre encastrée charge ponctuelle P à distance a
    P = force_magnitude
    a = force_x_pos
    x = mesh_x
    
    # Pour x <= a
    mask_before = x <= a
    displacement_y[mask_before] = (P / (6 * E_MODULUS * I_INERTIA)) * (x[mask_before]**2) * (3*a - x[mask_before])
    
    # Pour x > a (Déformation linéaire après le point de force)
    mask_after = x > a
    if np.any(mask_after):
        delta_a = (P * a**3) / (3 * E_MODULUS * I_INERTIA) # Flèche au point a
        theta_a = (P * a**2) / (2 * E_MODULUS * I_INERTIA) # Angle au point a
        displacement_y[mask_after] = delta_a + theta_a * (x[mask_after] - a)
    
    # Le déplacement est vers le bas (négatif selon l'axe y conventionnel, mais ici on traite la magnitude)
    displacement_y = -displacement_y 

    # 2. Calcul des Contraintes (Stress) de flexion
    # Sigma = - (Moment * y_local) / I
    # y_local est la distance par rapport à l'axe neutre (centre de la poutre)
    y_local = mesh_y - (HEIGHT / 2)
    
    moment = np.zeros_like(x)
    # Moment = P * (x - a) pour x < a (en convention statique, M(x) = P(x-a))
    # Mais amplitude du Moment = P * (a - x)
    moment[mask_before] = P * (a - x[mask_before])
    moment[mask_after] = 0 # Pas de moment après la charge (négligeant le poids propre)
    
    stress_sigma = -(moment * y_local) / I_INERTIA

    return displacement_y, stress_sigma

def generate_dataset():
    print(f"Génération de {NUM_SAMPLES} simulations...")
    
    data_inputs = [] # [x, y, force_pos, force_mag]
    data_targets = [] # [disp_y, stress]

    # Création de la grille de base (Maillage)
    xs = np.linspace(0, LENGTH, NODES_X)
    ys = np.linspace(0, HEIGHT, NODES_Y)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    mesh_x = mesh_x.flatten()
    mesh_y = mesh_y.flatten()
    
    # Normalisation des coordonnées pour le réseau de neurones (0-1)
    norm_x = mesh_x / LENGTH
    norm_y = mesh_y / HEIGHT

    for i in range(NUM_SAMPLES):
        # Paramètres aléatoires de la force
        f_pos = np.random.uniform(0.5, LENGTH) # Force appliquée quelque part
        f_mag = np.random.uniform(1000, 10000) # Force entre 1kN et 10kN
        
        # Calcul physique
        disp, stress = get_analytical_solution(f_pos, f_mag, mesh_x, mesh_y)
        
        # Préparation des tenseurs pour UN échantillon (graph)
        # Features par noeud : [Pos X, Pos Y, Force Pos, Force Mag]
        # On duplique les infos globales (Force) sur chaque noeud pour que le Transformer ait le contexte
        
        # Normalisation des inputs physiques pour aider le modèle
        norm_f_pos = f_pos / LENGTH
        norm_f_mag = (f_mag - 1000) / (9000) 
        
        node_features = np.stack([
            norm_x, 
            norm_y, 
            np.full_like(norm_x, norm_f_pos),
            np.full_like(norm_x, norm_f_mag)
        ], axis=1)
        
        # Targets normalisées (approximatif pour aider la convergence)
        # On divise par des constantes arbitraires proches des max attendus
        target_disp = disp / 0.1  # Max disp approx 10cm
        target_stress = stress / 50e6 # Max stress approx 50 MPa
        
        node_targets = np.stack([target_disp, target_stress], axis=1)
        
        data_inputs.append(node_features)
        data_targets.append(node_targets)

    # Conversion en Tensors PyTorch
    X = torch.tensor(np.array(data_inputs), dtype=torch.float32)
    Y = torch.tensor(np.array(data_targets), dtype=torch.float32)
    
    print(f"Dataset généré : Inputs {X.shape}, Targets {Y.shape}")
    torch.save({'X': X, 'Y': Y}, 'beam_dataset.pt')
    print("Fichier 'beam_dataset.pt' sauvegardé.")

if __name__ == "__main__":
    generate_dataset()