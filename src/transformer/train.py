import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

# --- ARCHITECTURE DU MODÈLE ---
class PhysicsAwareTransformer(nn.Module):
    def __init__(self, input_dim=4, model_dim=64, num_heads=4, num_layers=3, output_dim=2):
        super(PhysicsAwareTransformer, self).__init__()
        
        # Projection des features brutes vers l'espace latent du Transformer
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # Encodage positionnel (Appris ou Sinusoïdal - ici simple MLP appris)
        self.pos_encoder = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        # Bloc Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=128, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Tête de prédiction (Décodeur simple)
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim) # Prédiction [Deformation, Stress]
        )

    def forward(self, x):
        # x shape: [Batch, Num_Nodes, Features]
        
        # Embedding
        h = self.embedding(x)
        
        # Ajout d'information positionnelle (critique pour la géométrie)
        # Ici on ajoute l'input original transformé au vecteur latent
        pos_info = self.pos_encoder(x) 
        h = h + pos_info 
        
        # Passage dans le Transformer (Self-Attention sur tous les noeuds)
        # Le modèle apprend les relations physiques entre les points
        h = self.transformer(h)
        
        # Prédiction finale par noeud
        output = self.decoder(h)
        return output

# --- FONCTION D'ENTRAÎNEMENT ---
def train():
    # Configuration GPU (RTX 5070 Ti)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    # Chargement des données
    try:
        data = torch.load('beam_dataset.pt')
    except FileNotFoundError:
        print("Erreur: Lancez 'data_generator.py' d'abord !")
        return

    X, Y = data['X'], data['Y']
    
    # Split Train/Val (80% / 20%)
    split_idx = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split_idx], Y[:split_idx])
    val_ds = TensorDataset(X[split_idx:], Y[split_idx:])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # Initialisation du modèle
    model = PhysicsAwareTransformer().to(device)
    
    # Optimisation
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss() # Mean Squared Error pour la régression physique

    epochs = 50
    print("Début de l'entraînement...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")

    # Sauvegarde
    torch.save(model.state_dict(), 'mesh_transformer.pth')
    print("Modèle entraîné et sauvegardé sous 'mesh_transformer.pth'")

if __name__ == "__main__":
    train()