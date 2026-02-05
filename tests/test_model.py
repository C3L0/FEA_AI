import pytest
import torch

from fea_gnn.model import HybridPhysicsGNN

### Sadly those test didn't have been updated and the end of the project....


def test_model_output():
    # Create a minimal graph for the test
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.randn(2, 2)
    model = HybridPhysicsGNN(edge_index, edge_attr, hidden_dim=16, layers=2)

    x = torch.randn(1, 2, 5)  # [Batch, Nodes, Features]
    out = model(x)
    assert out.shape == (1, 2, 2)
