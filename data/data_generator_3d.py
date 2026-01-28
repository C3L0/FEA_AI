import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import dolfinx.io.gmsh as gmshio
import ufl
from petsc4py.PETSc import ScalarType
import pandas as pd

def create_heat_sink_mesh(comm, L, W, H_base, H_fin, N_fins, R_hole):
    """
    Crée un dissipateur thermique 3D.
    L, W : Longueur et largeur de la base
    H_base : Épaisseur de la base
    H_fin : Hauteur des ailettes
    N_fins : Nombre d'ailettes
    R_hole : Rayon des perforations dans les ailettes
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model
    model.add("HeatSink3D")
    occ = model.occ

    # 1. Création de la base
    base = occ.addBox(0, 0, 0, L, W, H_base)
    
    # 2. Création des ailettes avec perforations
    fin_thickness = (L / (2 * N_fins)) # Épaisseur proportionnelle
    fins_tags = []
    holes_tags = []

    for i in range(N_fins):
        x_pos = i * (L / N_fins) + fin_thickness/2
        # On crée l'ailette
        fin = occ.addBox(x_pos, 0, H_base, fin_thickness, W, H_fin)
        
        # On crée un trou cylindrique au milieu de l'ailette
        # Cylindre : (x, y, z, dx, dy, dz, rayon)
        hole = occ.addCylinder(x_pos - 0.1*fin_thickness, W/2, H_base + H_fin/2, 
                               fin_thickness * 1.2, 0, 0, R_hole)
        
        # On soustrait le trou de l'ailette
        fin_perforated, _ = occ.cut([(3, fin)], [(3, hole)])
        fins_tags.append(fin_perforated[0])

    # 3. Union de la base et des ailettes
    all_parts = [(3, base)] + fins_tags
    final_shape, _ = occ.fuse([all_parts[0]], all_parts[1:])
    
    occ.synchronize()

    # 4. Groupes Physiques et Maillage
    # On identifie le volume final
    volume_tag = final_shape[0][1]
    model.addPhysicalGroup(3, [volume_tag], tag=1)
    
    # Maillage Tetraédrique
    lc = H_fin / 5
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc / 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    model.mesh.generate(3)

    # Importation dans dolfinx
    mesh_data = gmshio.model_to_mesh(model, comm, 0, gdim=3)
    msh = mesh_data.mesh
    gmsh.finalize()
    return msh

def solve_elasticity_3d(msh, E, nu, P_load, comm):
    """Résout l'élasticité 3D sous une pression latérale P_load."""
    dim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", 1, (dim,)))

    # Propriétés matériau
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    def sigma(u):
        return 2.0 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(dim)

    # BC : Encastrement de la face inférieure (z=0)
    def bottom_face(x): return np.isclose(x[2], 0.0)
    f_bottom = mesh.locate_entities_boundary(msh, dim-1, bottom_face)
    bc = fem.dirichletbc(np.array([0, 0, 0], dtype=ScalarType), 
                         fem.locate_dofs_topological(V, dim-1, f_bottom), V)

    # Force : Pression sur les faces supérieures des ailettes (z=max)
    z_max = comm.allreduce(np.max(msh.geometry.x[:, 2]), op=MPI.MAX)
    def top_faces(x): return np.isclose(x[2], z_max)
    f_top = mesh.locate_entities_boundary(msh, dim-1, top_faces)
    tags = mesh.meshtags(msh, dim-1, f_top, np.full_like(f_top, 1))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    # On applique une pression verticale vers le bas
    L_form = ufl.dot(ufl.as_vector([0, 0, -P_load]), v) * ds(1)

    problem = LinearProblem(a, L_form, bcs=[bc], petsc_options_prefix="elasticity3d")
    uh = problem.solve()

    # Von Mises
    s = sigma(uh) - (1./3)*ufl.tr(sigma(uh))*ufl.Identity(dim)
    vm_expr = ufl.sqrt(3./2 * ufl.inner(s, s))
    W = fem.functionspace(msh, ("Lagrange", 1))
    p, w = ufl.TrialFunction(W), ufl.TestFunction(W)
    proj = LinearProblem(ufl.inner(p, w)*ufl.dx, ufl.inner(vm_expr, w)*ufl.dx, petsc_options_prefix="vm")
    vm_field = proj.solve()

    return uh, vm_field

def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    n_samples_total = 250 # On commence par 100 car la 3D est gourmande
    n_samples_local = n_samples_total // size
    rng = np.random.default_rng(42 + rank)
    
    local_data = []

    for i in range(n_samples_local):
        # Paramètres aléatoires
        L, W = 0.1, 0.1 # Base fixe de 10cm x 10cm
        H_base = rng.uniform(0.005, 0.015)
        H_fin = rng.uniform(0.03, 0.08)
        N_fins = int(rng.integers(3, 8))
        R_hole = rng.uniform(0.002, 0.008)
        E, nu = rng.uniform(70e9, 210e9), 0.3
        P_load = rng.uniform(1e5, 1e6)

        try:
            msh = create_heat_sink_mesh(MPI.COMM_SELF, L, W, H_base, H_fin, N_fins, R_hole)
            uh, vm = solve_elasticity_3d(msh, E, nu, P_load, comm)
            
            u_max = np.max(np.linalg.norm(uh.x.array.reshape(-1, 3), axis=1))
            sig_max = np.max(vm.x.array)
            
            local_data.append([H_base, H_fin, N_fins, R_hole, E, P_load, u_max, sig_max])
            if rank == 0: print(f"Échantillon {i*size} terminé")
        except Exception as e:
            continue

    all_data = comm.gather(local_data, root=0)
    if rank == 0:
        flat = [item for sublist in all_data for item in sublist]
        df = pd.DataFrame(flat, columns=["H_base", "H_fin", "N_fins", "R_hole", "E", "P_load", "u_max", "sig_max"])
        df.to_csv("heatsink_3d_dataset.csv", index=False)

if __name__ == "__main__":
    main()
