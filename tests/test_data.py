import numpy as np
import pytest
import torch

from fea_gnn.data_loader import CantileverMeshDataset


def test_dataset_structure():
    """Checks if the dataset returns the correct data types and shapes."""
    # Initialize small dataset
    ds = CantileverMeshDataset(
        num_samples=5,
        nx=10,
        ny=2,
        length=1.0,
        height=0.1,
        E_range=[200, 210],
        nu_range=[0.3, 0.3],
    )

    # Get first sample
    features, labels = ds[0]

    # Check types
    assert isinstance(features, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

    # Check shapes
    # Features should be [Nodes, 5] (Fx, Fy, Fix, E, nu)
    assert features.shape[1] == 5
    # Labels should be [Nodes, 2] (u, v)
    assert labels.shape[1] == 2


def test_material_normalization():
    """
    CRITICAL: Tests if Young's Modulus is normalized to approx 1.0
    If we feed E=210 GPa, the model should receive ~1.0, not 210e9.
    """
    ds = CantileverMeshDataset(
        num_samples=1,
        nx=5,
        ny=2,
        length=1.0,
        height=0.1,
        E_range=[210.0, 210.0],  # Force E to be exactly 210
        nu_range=[0.3, 0.3],
    )

    features, _ = ds[0]

    # Extract E channel (Index 3)
    e_values = features[:, 3]

    # Since logic is E / 210.0, result should be exactly 1.0
    assert torch.allclose(
        e_values, torch.tensor(1.0), atol=1e-5
    ), f"Normalization failed! Expected 1.0, got {e_values.mean()}"
