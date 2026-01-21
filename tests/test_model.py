import torch
import pytest

# Correct : fea_gnn est reconnu comme package grâce au pyproject.toml
from fea_gnn.model import HybridPhysicsGNN


def test_model_output():
    # Création d'un graphe minimal pour le test
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.randn(2, 2)
    model = HybridPhysicsGNN(edge_index, edge_attr, hidden_dim=16, layers=2)

    x = torch.randn(1, 2, 5)  # [Batch, Nodes, Features]
    out = model(x)
    assert out.shape == (1, 2, 2)
