from data.data_pipeline import PlateHoleDataset


def get_dataset(root="."):
    """Charge le dataset de plaques trouées généré précédemment."""
    return PlateHoleDataset(root=root)


# import numpy as np
# import torch
# from torch.utils.data import Dataset
#
#
# class CantileverMeshDataset(Dataset):
#     def __init__(self, num_samples, nx, ny, length, height, E_range, nu_range):
#         self.num_samples = num_samples
#         self.nx = nx
#         self.ny = ny
#         self.length = length
#         self.height = height
#
#         # 1. Create Grid Coordinates
#         x = np.linspace(0, length, nx)
#         y = np.linspace(0, height, ny)
#         xv, yv = np.meshgrid(x, y)
#         self.coords = np.stack([xv.flatten(), yv.flatten()], axis=1)
#         self.num_nodes = nx * ny
#
#         # 2. Create Edges (Topology)
#         self.edge_index = self._create_edges()
#
#         # 3. Create Edge Attributes (Relative distances dx, dy)
#         self.edge_attr = self._create_edge_attr()
#
#         # 4. Generate Samples
#         self.samples_x = []  # Node features [Fx, Fy, isFixed, E, nu]
#         self.samples_y = []  # Target displacements [u, v]
#
#         for _ in range(num_samples):
#             # Random Material Properties for this specific beam
#             E = np.random.uniform(E_range[0], E_range[1])  # Young's Modulus
#             nu = np.random.uniform(nu_range[0], nu_range[1])  # Poisson's Ratio
#
#             # Random Force at the tip
#             force_mag = np.random.uniform(-100, -10)
#
#             # Features and Labels for this simulation
#             feat, label = self._generate_mock_fea(force_mag, E, nu)
#             self.samples_x.append(feat)
#             self.samples_y.append(label)
#
#     def _create_edges(self):
#         edges = []
#         for i in range(self.ny):
#             for j in range(self.nx):
#                 node = i * self.nx + j
#                 if j < self.nx - 1:  # Right neighbor
#                     edges.append([node, node + 1])
#                 if i < self.ny - 1:  # Top neighbor
#                     edges.append([node, node + self.nx])
#                 if j < self.nx - 1 and i < self.ny - 1:  # Diagonal
#                     edges.append([node, node + self.nx + 1])
#
#         edge_index = np.array(edges).T
#         # Make it undirected (bi-directional)
#         edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
#         return torch.tensor(edge_index, dtype=torch.long)
#
#     def _create_edge_attr(self):
#         row, col = self.edge_index
#         pos = self.coords
#         diff = pos[col] - pos[row]
#         return torch.tensor(diff, dtype=torch.float)
#
#     def _generate_mock_fea(self, force_mag, E, nu):
#         """
#         Calculates displacements based on Beam Theory:
#         v = (F * x^2 * (3L - x)) / (6 * E * I)
#         """
#         nodes_x = self.coords[:, 0]
#         nodes_y = self.coords[:, 1]
#
#         # Moment of Inertia (simplified)
#         I = (1.0 * self.height**3) / 12.0
#
#         # Vertical Displacement (v) - depends on 1/E
#         # We add some noise to make it "FEA-like"
#         v = (force_mag * nodes_x**2 * (3 * self.length - nodes_x)) / (6 * E * I)
#
#         # Horizontal Displacement (u) - influenced by Poisson's ratio (thinning)
#         u = (
#             -nu
#             * (nodes_y - self.height / 2)
#             * (force_mag * nodes_x * (self.length - nodes_x))
#             / (E * I)
#         )
#
#         # Build Node Features [Fx, Fy, isFixed, E, nu]
#         features = np.zeros((self.num_nodes, 5))
#         features[:, 2] = (nodes_x == 0).astype(float)  # isFixed
#         features[:, 3] = E / 210.0  # Normalized E (relative to steel)
#         features[:, 4] = nu  # Poisson's ratio
#
#         # Apply force only at the tip nodes (right side)
#         tip_mask = nodes_x == self.length
#         features[tip_mask, 1] = force_mag / np.sum(tip_mask)
#
#         displacements = np.stack([u, v], axis=1)
#
#         return torch.tensor(features, dtype=torch.float), torch.tensor(
#             displacements, dtype=torch.float
#         )
#
#     def __len__(self):
#         return self.num_samples
#
#     def __getitem__(self, idx):
#         return self.samples_x[idx], self.samples_y[idx]
