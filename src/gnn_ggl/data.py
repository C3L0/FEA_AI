import numpy as np
import torch
from torch.utils.data import Dataset


class CantileverMeshDataset(Dataset):
    """Generates a graph-based mesh for a 2D beam."""

    def __init__(self, nx=25, ny=5, length=2.0, height=0.5, num_samples=1000):
        self.nx, self.ny = nx, ny
        self.length, self.height = length, height

        # Static Mesh Geometry
        x = np.linspace(0, length, nx)
        y = np.linspace(0, height, ny)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.stack([xx.flatten(), yy.flatten()], axis=1)

        # Topology: Connectivity (Edges)
        edges = []
        for i in range(ny):
            for j in range(nx):
                n = i * nx + j
                if j < nx - 1:
                    edges += [[n, n + 1], [n + 1, n]]  # Horizontal
                if i < ny - 1:
                    edges += [[n, n + nx], [n + nx, n]]  # Vertical
                if i < ny - 1 and j < nx - 1:
                    edges += [[n, n + nx + 1], [n + nx + 1, n]]  # Diag

        self.edge_index = torch.tensor(edges, dtype=torch.long).t()

        # IMPROVEMENT: Pre-calculate Relative Distance (Edge Attributes)
        pos = torch.tensor(self.coords, dtype=torch.float32)
        self.edge_attr = pos[self.edge_index[1]] - pos[self.edge_index[0]]

        # Generate target samples
        self.samples = []
        for _ in range(num_samples):
            load_p = np.random.uniform(-2, -15)
            # Node Features: [Fx, Fy, isFixed]
            feat = np.zeros((nx * ny, 3))
            feat[self.coords[:, 0] < 0.01, 2] = 1.0  # Fixed left
            right_mask = self.coords[:, 0] > (length - 0.01)
            feat[right_mask, 1] = load_p / np.sum(right_mask)  # Distribute Fy

            # Analytical Ground Truth
            gt = self._analytical_solution(load_p)
            self.samples.append(
                (
                    torch.tensor(feat, dtype=torch.float32),
                    torch.tensor(gt, dtype=torch.float32),
                )
            )

    def _analytical_solution(self, p, e=100, i_mom=0.01):
        x, y = self.coords[:, 0], self.coords[:, 1]
        v = (p * x**2) / (6 * e * i_mom) * (3 * self.length - x)
        dv_dx = (p * x) / (2 * e * i_mom) * (2 * self.length - x)
        u = -(y - self.height / 2) * dv_dx
        return np.stack([u, v], axis=1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
