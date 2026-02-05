import os

# On autorise les imports mixtes pour FEniCSx avant tout import dolfinx
os.environ["DOLFINX_ALLOW_USER_SITE_IMPORTS"] = "1"

import dolfinx.io.gmsh as gmshio
import gmsh
import numpy as np
import pandas as pd
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType


def create_plate_with_hole_mesh(comm, L, H, R, res_factor):
    """
    Create the meshing with variable density
    High res_factor élevé = fine meshing
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model
    model.add("Plate_with_Hole")

    # Caracteristic size depends of an random factor
    lc = H / res_factor

    rect = model.occ.addRectangle(0, 0, 0, L, H)
    circle = model.occ.addDisk(L / 2, H / 2, 0, R, R)
    out, _ = model.occ.cut([(2, rect)], [(2, circle)])
    model.occ.synchronize()
    surface_tag = out[0][1]
    model.addPhysicalGroup(2, [surface_tag], tag=1)

    # Refinement
    model.mesh.field.add("Distance", 1)
    entities = model.getEntities(1)
    model.mesh.field.setNumbers(1, "CurvesList", [e[1] for e in entities])
    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "InField", 1)
    model.mesh.field.setNumber(2, "SizeMin", lc / 3)
    model.mesh.field.setNumber(2, "SizeMax", lc)
    model.mesh.field.setNumber(2, "DistMin", R * 0.5)
    model.mesh.field.setNumber(2, "DistMax", R * 2)
    model.mesh.field.setAsBackgroundMesh(2)
    model.mesh.generate(2)

    mesh_data = gmshio.model_to_mesh(model, comm, 0, gdim=2)
    msh = mesh_data.mesh
    gmsh.finalize()
    return msh


def solve_elasticity(msh, E, nu, F):
    dim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", 1, (dim,)))
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(dim)

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    f_left = mesh.locate_entities_boundary(msh, dim - 1, left_boundary)
    bc_left = fem.dirichletbc(
        np.array([0, 0], dtype=ScalarType),
        fem.locate_dofs_topological(V, dim - 1, f_left),
        V,
    )

    def right_boundary(x):
        return np.isclose(x[0], np.max(msh.geometry.x[:, 0]))

    f_right = mesh.locate_entities_boundary(msh, dim - 1, right_boundary)
    tags = mesh.meshtags(msh, dim - 1, f_right, np.full_like(f_right, 1))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L_form = ufl.dot(ufl.as_vector([F, 0.0]), v) * ds(1)

    # Stability options
    problem = LinearProblem(a, L_form, bcs=[bc_left], petsc_options_prefix="elasticity")
    uh = problem.solve()
    return uh, V


def extract_simulation_data(sim_id, msh, V, uh, E, nu, F):
    dof_coords = V.tabulate_dof_coordinates()
    u_array = uh.x.array if hasattr(uh.x, "array") else uh.x
    u_values = np.real(u_array).reshape(-1, 2)

    x_nodes = dof_coords[:, 0]
    y_nodes = dof_coords[:, 1]
    num_nodes = len(x_nodes)

    load_x = np.zeros(num_nodes)
    right_mask = np.isclose(x_nodes, np.max(x_nodes), atol=1e-3)
    load_x[right_mask] = F
    is_fixed = np.isclose(x_nodes, 0.0, atol=1e-3).astype(int)

    node_rows = []
    for i in range(num_nodes):
        node_rows.append(
            {
                "SimulationID": sim_id,
                "NodeID": i,
                "x": x_nodes[i],
                "y": y_nodes[i],
                "E": E,
                "nu": nu,
                "Fx": load_x[i],
                "Fy": 0.0,
                "isFixed": is_fixed[i],
                "ux": u_values[i, 0],
                "uy": u_values[i, 1],
            }
        )

    dofmap_obj = V.dofmap.list
    cells = dofmap_obj.array if hasattr(dofmap_obj, "array") else dofmap_obj
    cells = cells.reshape(-1, 3)

    connectivity_rows = []
    for cell in cells:
        connectivity_rows.append(
            {"SimulationID": sim_id, "n1": cell[0], "n2": cell[1], "n3": cell[2]}
        )

    return node_rows, connectivity_rows


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # Nombre de simulations (Ajustez selon vos besoins, ex: 250 ou 500)
    n_samples_total = 600
    n_samples_local = n_samples_total // size
    rng = np.random.default_rng(42 + rank)

    local_nodes = []
    local_connectivity = []

    if rank == 0:
        print(f"Generation of {n_samples_total} simulations on {size} cores")

    for i in range(n_samples_local):
        sim_id = rank * n_samples_local + i

        # Variable geometry
        L, H = rng.uniform(1.5, 2.5), rng.uniform(0.8, 1.2)
        R = rng.uniform(0.1, 0.3) * H

        # Variable materials
        E, nu = rng.uniform(10e9, 210e9), rng.uniform(0.2, 0.4)
        F = rng.uniform(1e5, 1e7)

        # Variable meshing density
        res_factor = rng.uniform(8.0, 25.0)

        try:
            msh = create_plate_with_hole_mesh(MPI.COMM_SELF, L, H, R, res_factor)
            uh, V = solve_elasticity(msh, E, nu, F)

            nodes, topo = extract_simulation_data(sim_id, msh, V, uh, E, nu, F)
            local_nodes.extend(nodes)
            local_connectivity.extend(topo)

            if rank == 0:
                print(f"Sim {sim_id} OK ({len(nodes)} nodes)")

        except Exception as e:
            print(f"Error Sim {sim_id}: {e}")

    all_nodes = comm.gather(local_nodes, root=0)
    all_topo = comm.gather(local_connectivity, root=0)

    if rank == 0:
        flat_nodes = [item for sublist in all_nodes for item in sublist]
        flat_topo = [item for sublist in all_topo for item in sublist]

        output_dir = "raw"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pd.DataFrame(flat_nodes).to_csv(f"{output_dir}/db.csv", index=False)
        pd.DataFrame(flat_topo).to_csv(f"{output_dir}/connectivity.csv", index=False)

        print(
            f"\nSuccess ! Fiels generated in '{output_dir}/' :"
            f"\n -> db.csv ({len(flat_nodes)} lines)"
            f"\n -> connectivity.csv ({len(flat_topo)} triangles)"
        )


if __name__ == "__main__":
    main()
