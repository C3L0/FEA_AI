import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

from src.gnn_ggl.data import CantileverMeshDataset
from src.gnn_ggl.model import SolidGNN
from src.gnn_ggl.evaluation import visualize_gnn_results 

def run_simulation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataset = CantileverMeshDataset(num_samples=800)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = SolidGNN(dataset.edge_index.to(device), dataset.edge_attr.to(device)).to(
        device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # Training Loop
    model.train()
    for epoch in range(150):
        epoch_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {epoch_loss / len(loader):.6f}")

    visualize_gnn_results(model, dataset, device, sample_idx=5)

     # Save the model
    if not os.path.exists("src/gnn_ggl/model"):
        os.makedirs("src/gnn_ggl/model")
    torch.save(model.state_state_dict(), "src/gnn_ggl/model/gnn_v2.pth")
    print("Training complete and model saved.")

    # Simple Visual Eval
    # model.eval()
    # with torch.no_grad():
    #     test_x, test_y = dataset[0]
    #     pred_y = model(test_x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

    # plt.scatter(dataset.coords[:, 0], dataset.coords[:, 1], c="gray", alpha=0.2)
    # plt.scatter(
    #     dataset.coords[:, 0] + pred_y[:, 0] * 5,
    #     dataset.coords[:, 1] + pred_y[:, 1] * 5,
    #     c="red",
    #     label="GNN Pred",
    # )
    # plt.legend()
    # plt.title("GNN Result (Amplified 5x)")
    # plt.show()


if __name__ == "__main__":
    run_simulation()
