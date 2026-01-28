from os import wait

import torch

from data.data_pipeline import PlateHoleDataset

# Charger le dataset proprement via la classe PyG
dataset = PlateHoleDataset(root="data/")

print(f"Nombre de simulations réelles : {len(dataset)}")

# On regarde la première simulation
sim0 = dataset[0]
print(f"Structure Simu 0 : {sim0}")

# Analyse du déplacement vertical (uy est à l'index 1)
uy = sim0.y[:, 1]

print("-" * 30)
print(f"Stats Déplacement Vertical de la Simu 0 :")
print(f"Max:  {uy.max().item():.6f}")
print(f"Mean: {uy.mean().item():.8f}")

# Vérification du scaling
if uy.max().item() > 10.0:
    print("Le scaling x1000 semble appliqué !")
else:
    print("Le scaling n'est toujours pas visible (Max < 10).")

ux = sim0.x[:, 1]

print("-" * 30)
print(f"Stats Déplacement Vertical de la Simu 0 :")
print(f"Max:  {ux.max().item():.6f}")
print(f"Mean: {ux.mean().item():.8f}")

# Vérification du scaling
if ux.max().item() > 10.0:
    print("Le scaling x1000 semble appliqué !")
else:
    print("Le scaling n'est toujours pas visible (Max < 10).")
