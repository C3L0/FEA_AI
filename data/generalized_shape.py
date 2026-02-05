import os

import dolfinx.io.gmsh as gmshio
import gmsh
import numpy as np
import pandas as pd
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# Ensure mixed imports work
os.environ["DOLFINX_ALLOW_USER_SITE_IMPORTS"] = "1"


def create_full_plate(comm, L, H, res_factor=15):
    """Generates a simple rectangular plate without holes"""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model
    model.add("Full_Plate")

    lc = H / res_factor

    # Simple Rectangle
    rect = model.occ.addRectangle(0, 0, 0, L, H)
    model.occ.synchronize()

    model.addPhysicalGroup(2, [rect], tag=1)

    # Uniform meshing
    model.mesh.setSize(model.getEntities(0), lc)
    model.mesh.generate(2)

    mesh_data = gmshio.model_to_mesh(model, comm, 0, gdim=2)
    gmsh.finalize()
    return mesh_data.mesh


def create_double_hole_plate(comm, L, H, R, res_factor=15):
    """Generates a plate with TWO holes"""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model
    model.add("Double_Hole")

    lc = H / res_factor

    rect = model.occ.addRectangle(0, 0, 0, L, H)

    # Hole 1
    c1 = model.occ.addDisk(L / 3, H / 2, 0, R, R)
    # Hole 2
    c2 = model.occ.addDisk(2 * L / 3, H / 2, 0, R, R)

    # Cut both
    out, _ = model.occ.cut([(2, rect)], [(2, c1), (2, c2)])
    model.occ.synchronize()

    surface_tag = out[0][1]
    model.addPhysicalGroup(2, [surface_tag], tag=1)

    # Refinement around holes
    model.mesh.setSize(model.getEntities(0), lc)
    model.mesh.generate(2)

    mesh_data = gmshio.model_to_mesh(model, comm, 0, gdim=2)
    gmsh.finalize()
    return mesh_data.mesh


def solve_elasticity(msh, E, nu, F):
    dim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", 1, (dim,)))
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(dim)

    # Fixed Left
    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    f_left = mesh.locate_entities_boundary(msh, dim - 1, left_boundary)
    bc_left = fem.dirichletbc(
        np.array([0, 0], dtype=ScalarType),
        fem.locate_dofs_topological(V, dim - 1, f_left),
        V,
    )

    # Force Right
    def right_boundary(x):
        return np.isclose(x[0], np.max(msh.geometry.x[:, 0]))

    f_right = mesh.locate_entities_boundary(msh, dim - 1, right_boundary)
    tags = mesh.meshtags(msh, dim - 1, f_right, np.full_like(f_right, 1))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L_form = ufl.dot(ufl.as_vector([F, 0.0]), v) * ds(1)

    problem = LinearProblem(
        a, L_form, bcs=[bc_left], petsc_options_prefix="test_elast_"
    )
    uh = problem.solve()
    return uh, V


def extract_data(sim_id, msh, V, uh, E, nu, F):
    dof_coords = V.tabulate_dof_coordinates()
    u_array = uh.x.array if hasattr(uh.x, "array") else uh.x
    u_values = np.real(u_array).reshape(-1, 2)
    x, y = dof_coords[:, 0], dof_coords[:, 1]

    # Features
    load_x = np.zeros(len(x))
    load_x[np.isclose(x, np.max(x), atol=1e-3)] = F
    is_fixed = np.isclose(x, 0.0, atol=1e-3).astype(int)

    nodes = []
    for i in range(len(x)):
        nodes.append(
            {
                "SimulationID": sim_id,
                "NodeID": i,
                "x": x[i],
                "y": y[i],
                "E": E,
                "nu": nu,
                "Fx": load_x[i],
                "Fy": 0.0,
                "isFixed": is_fixed[i],
                "ux": u_values[i, 0],
                "uy": u_values[i, 1],
            }
        )

    # Topology
    dofmap = V.dofmap.list
    cells = dofmap.array if hasattr(dofmap, "array") else dofmap
    cells = cells.reshape(-1, 3)
    topo = [{"SimulationID": sim_id, "n1": c[0], "n2": c[1], "n3": c[2]} for c in cells]

    return nodes, topo


def main():
    # Setup folders
    base_dir = "data/generalization"
    os.makedirs(f"{base_dir}/full_plate", exist_ok=True)
    os.makedirs(f"{base_dir}/double_hole", exist_ok=True)

    comm = MPI.COMM_WORLD

    # Shared Parameters
    L, H = 2.0, 0.5
    E, nu = 210e9, 0.3
    F = 1e6

    print("Generating Case 1: Full Plate (No Hole)")
    msh_full = create_full_plate(comm, L, H)
    uh_full, V_full = solve_elasticity(msh_full, E, nu, F)
    nodes_full, topo_full = extract_data(101, msh_full, V_full, uh_full, E, nu, F)

    pd.DataFrame(nodes_full).to_csv(f"{base_dir}/full_plate/db.csv", index=False)
    pd.DataFrame(topo_full).to_csv(
        f"{base_dir}/full_plate/connectivity.csv", index=False
    )

    print("Generating Case 2: Double Hole")
    msh_double = create_double_hole_plate(comm, L, H, R=0.08)
    uh_double, V_double = solve_elasticity(msh_double, E, nu, F)
    nodes_double, topo_double = extract_data(
        102, msh_double, V_double, uh_double, E, nu, F
    )

    pd.DataFrame(nodes_double).to_csv(f"{base_dir}/double_hole/db.csv", index=False)
    pd.DataFrame(topo_double).to_csv(
        f"{base_dir}/double_hole/connectivity.csv", index=False
    )

    print("\nDONE! Data saved in 'data/generalization/'")


if __name__ == "__main__":
    main()
