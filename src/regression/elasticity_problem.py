from mpi4py import MPI
from petsc4py.PETSc import ScalarType

import numpy as np
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem


def epsilon(u):
    """Déformation symétrique ε(u) = 1/2 (∇u + ∇u^T)."""
    return ufl.sym(ufl.grad(u))


def sigma(u, E, nu, dim):
    """Contrainte de Cauchy pour élasticité linéaire isotrope."""
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(dim)


def generate_sample(comm, L, H, E, nu, F):
    """
    Génère une simulation FEA pour une poutre de longueur L, hauteur H,
    module de Young E, Poisson nu, force F appliquée à droite.
    Retourne u_max, sigma_vm_max.
    """
    rank = comm.rank
    tdim = 2

    # --- 1) Maillage ---
    msh = mesh.create_rectangle(
        comm=comm,
        points=((0.0, 0.0), (L, H)),
        n=(40, 10),
        cell_type=mesh.CellType.triangle,
    )

    dim = msh.geometry.dim  # 2 en 2D

    # Espace de fonctions vectoriel pour le déplacement u (2 composantes)
    # ⚠️ NOTE : le 3e argument est un SHAPE = (dim,), pas juste dim
    V = fem.functionspace(msh, ("Lagrange", 1, (dim,)))

    # --- 2) Conditions aux limites : encastrement à gauche (x = 0) ---
    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    facets_left = mesh.locate_entities_boundary(
        msh, dim=tdim - 1, marker=left_boundary
    )
    dofs_left = fem.locate_dofs_topological(V, entity_dim=tdim - 1, entities=facets_left)

    zero_vec = np.array((0.0, 0.0), dtype=ScalarType)
    bc_left = fem.dirichletbc(zero_vec, dofs_left, V)

    # --- 3) Traction sur le bord droit (x = L) ---
    def right_boundary(x):
        return np.isclose(x[0], L)

    facets_right = mesh.locate_entities_boundary(
        msh, dim=tdim - 1, marker=right_boundary
    )

    facet_indices = facets_right
    facet_values = np.full_like(facet_indices, 1, dtype=np.int32)
    facet_tags = mesh.meshtags(msh, tdim - 1, facet_indices, facet_values)

    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    # Force totale F vers le bas, répartie uniformément le long du bord droit (longueur H)
    traction_value = F / H
    t_vec = ufl.as_vector((ScalarType(0.0), ScalarType(-traction_value)))

    # --- 4) Formulation variationnelle de l'élasticité ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    sigma_u = sigma(u, E, nu, dim)
    a = ufl.inner(sigma_u, epsilon(v)) * ufl.dx
    L_form = ufl.dot(t_vec, v) * ds(1)

    problem = LinearProblem(
        a,
        L_form,
        bcs=[bc_left],
        petsc_options_prefix="elasticity_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
        }
    )

    uh = problem.solve()  # solution du déplacement

    # --- 5) Calcul de u_max (norme de déplacement max) ---
    u_values = uh.x.array.reshape((-1, dim))  # (N_dofs, 2)
    disp_norm = np.linalg.norm(u_values, axis=1)
    local_u_max = float(disp_norm.max()) if disp_norm.size > 0 else 0.0
    u_max = comm.allreduce(local_u_max, op=MPI.MAX)

    # --- 6) Calcul de sigma_vm_max (von Mises) ---
    sigma_uh = sigma(uh, E, nu, dim)
    sx = sigma_uh[0, 0]
    sy = sigma_uh[1, 1]
    txy = sigma_uh[0, 1]

    sigma_vm_expr = ufl.sqrt(sx**2 + sy**2 - sx * sy + 3.0 * txy**2)

    # Projection sur un espace scalaire P1
    W = fem.functionspace(msh, ("Lagrange", 1))
    p = ufl.TrialFunction(W)
    w = ufl.TestFunction(W)

    a_proj = ufl.inner(p, w) * ufl.dx
    L_proj = ufl.inner(sigma_vm_expr, w) * ufl.dx

    proj_problem = LinearProblem(
        a_proj,
        L_proj,
        petsc_options_prefix="proj_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
        },
    )

    sigma_vm = proj_problem.solve()

    sigma_values = sigma_vm.x.array
    local_sigma_max = float(sigma_values.max()) if sigma_values.size > 0 else 0.0
    sigma_vm_max = comm.allreduce(local_sigma_max, op=MPI.MAX)

    if rank == 0:
        print(
            f"L={L:.2f}, H={H:.3f}, E={E:.2e}, nu={nu:.2f}, F={F:.1f} "
            f"-> u_max={u_max:.6e}, sigma_vm_max={sigma_vm_max:.6e}"
        )

    return u_max, sigma_vm_max


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    n_samples = 5000
    X = []
    y = []

    rng = np.random.default_rng(42)

    for _ in range(n_samples):
        L = float(rng.uniform(0.5, 2.0))
        H = float(rng.uniform(0.05, 0.2))
        E = float(rng.uniform(70e9, 210e9))
        nu = float(rng.uniform(0.25, 0.35))
        F = float(rng.uniform(500.0, 5000.0))

        u_max, sigma_vm_max = generate_sample(comm, L, H, E, nu, F)

        if rank == 0:
            X.append([L, H, E, nu, F])
            y.append([u_max, sigma_vm_max])

    if rank == 0:
        X = np.array(X)
        y = np.array(y)
        dataset = np.hstack([X, y])

        header = "L,H,E,nu,F,u_max,sigma_vm_max"
        np.savetxt(
            "elasticity_beam_dataset.csv",
            dataset,
            delimiter=",",
            header=header,
            comments="",
        )
        print("\nDataset saved to elasticity_beam_dataset.csv")


if __name__ == "__main__":
    main()
