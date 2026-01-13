import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement et Préparation des données
# ----------------------------------------
df = pd.read_csv('elasticity_beam_dataset.csv')

# Séparation Entrées (Inputs) / Sorties (Targets)
inputs = df[['L', 'H', 'E', 'nu', 'F']].values
targets = df[['u_max', 'sigma_vm_max']].values

# Split Train/Test (80% entrainement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# IMPORTANT: Normalisation (StandardScaler)
# On met les données à une moyenne de 0 et un écart-type de 1
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Conversion en Tenseurs PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

# Création des DataLoaders pour l'entrainement par batch
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. Définition du Modèle (Réseau de Neurones)
# --------------------------------------------
class FEMPredictor(nn.Module):
    def __init__(self):
        super(FEMPredictor, self).__init__()
        # 5 entrées -> Couches cachées -> 2 sorties
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Sortie: [u_max, sigma_vm_max]
        )

    def forward(self, x):
        return self.net(x)

model = FEMPredictor()

# 3. Entrainement
# ---------------
criterion = nn.MSELoss()  # Mean Squared Error pour la régression
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
loss_history = []

print("Début de l'entrainement...")
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()         # Reset gradients
        outputs = model(batch_X)      # Prédiction
        loss = criterion(outputs, batch_y) # Calcul de l'erreur
        loss.backward()               # Backpropagation
        optimizer.step()              # Mise à jour des poids
        epoch_loss += loss.item()
    
    loss_history.append(epoch_loss / len(train_loader))
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss_history[-1]:.6f}")

# 4. Évaluation et Visualisation
# ------------------------------
model.eval()
with torch.no_grad():
    # Prédiction sur les données de test
    predictions_scaled = model(X_test_tensor)
    # On inverse la normalisation pour retrouver les vraies unités physiques
    predictions = scaler_y.inverse_transform(predictions_scaled.numpy())

# Calcul du score R2
mse = np.mean((y_test - predictions)**2)
print(f"\nErreur Quadratique Moyenne (MSE) sur le test set: {mse:.4e}")

# Visualisation des résultats pour u_max
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title('Courbe d\'apprentissage (Loss)')
plt.xlabel('Epochs')

plt.subplot(1, 2, 2)
plt.scatter(y_test[:, 0], predictions[:, 0], alpha=0.6, color='b')
plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'r--')
plt.title('Réalité vs Prédiction (u_max)')
plt.xlabel('Vrai u_max')
plt.ylabel('Prédit u_max')
plt.tight_layout()
plt.show()