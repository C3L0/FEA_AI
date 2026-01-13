import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
import matplotlib.pyplot as plt
import os

# --- Configuration ---
BATCH_SIZE = 32
HIDDEN_CHANNELS = 64
EPOCHS = 500
LR = 0.001
MODEL_PATH = 'wood_gnn_model.pth'
DATASET_PATH = 'wood_fem_dataset.pt'

# --- Définition du Modèle (Identique pour la compatibilité) ---
class WoodStressGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WoodStressGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, HIDDEN_CHANNELS)
        self.conv2 = SAGEConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv3 = SAGEConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.fc = torch.nn.Linear(HIDDEN_CHANNELS, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.fc(x)
        return x

def train():
    # 1. Vérification du dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Erreur : '{DATASET_PATH}' introuvable. Lancez 'data_generation.py' d'abord.")
        return

    dataset = torch.load(DATASET_PATH, weights_only=False)
    
    # Split Train/Test (80/20)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialisation du modèle
    model = WoodStressGNN(in_channels=4, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    # 3. Chargement du modèle existant si disponible
    if os.path.exists(MODEL_PATH):
        print(f"--- Modèle trouvé : Chargement de '{MODEL_PATH}' ---")
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print("Poids chargés avec succès. Reprise de l'entraînement...")
        except Exception as e:
            print(f"Erreur lors du chargement : {e}. On repart de zéro.")
    else:
        print("--- Aucun modèle pré-entraîné trouvé. Nouvel entraînement ---")

    # 4. Boucle d'entraînement
    losses = []
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d}/{EPOCHS}, Loss (MSE): {avg_loss:.8f}')
            
    # 5. Sauvegarde finale
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Entraînement terminé. Modèle sauvegardé sous '{MODEL_PATH}'.")
    
    # Visualisation de la perte
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Train Loss')
    plt.yscale('log') # Echelle log pour mieux voir la convergence
    plt.title("Évolution de la perte (Mean Squared Error)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig("training_loss.png")
    print("Graphique 'training_loss.png' mis à jour.")

if __name__ == "__main__":
    train()