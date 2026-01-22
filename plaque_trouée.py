import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
import dolfinx.io.gmsh as gmshio
import ufl
from petsc4py.PETSc import ScalarType
import pandas as pd

def create_plate_with_hole_mesh(comm, L, H, R):
    """Crée un maillage de plaque trouée avec GMSH et définit les groupes physiques."""
    gmsh.initialize()
    # Désactiver les logs GMSH dans le terminal pour y voir plus clair
    gmsh.option.setNumber("General.Terminal", 0) 
    
    model = gmsh.model
    model.add("Plate_with_Hole")
    
    lc = H / 15 
    
    # 1. Création de la géométrie
    rect = model.occ.addRectangle(0, 0, 0, L, H)
    circle = model.occ.addDisk(L/2, H/2, 0, R, R)
    # Note: cut retourne une liste de tuples [(dimension, tag)]
    out, out_map = model.occ.cut([(2, rect)], [(2, circle)])
    
    model.occ.synchronize()
    
    # On doit dire à GMSH que la surface résultante est un domaine physique (dim=2)
    # On récupère le tag de la surface après la découpe
    surface_tag = out[0][1]
    model.addPhysicalGroup(2, [surface_tag], tag=1)
    model.setPhysicalName(2, 1, "Plate_Surface")
    # ------------------------------------------
    
    # Raffinement local autour du trou
    model.mesh.field.add("Distance", 1)
    # Après un cut, les IDs des lignes peuvent changer, on utilise une méthode robuste
    entities = model.getEntities(1) # Récupère toutes les lignes
    model.mesh.field.setNumbers(1, "CurvesList", [e[1] for e in entities])
    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "InField", 1)
    model.mesh.field.setNumber(2, "SizeMin", lc / 3)
    model.mesh.field.setNumber(2, "SizeMax", lc)
    model.mesh.field.setNumber(2, "DistMin", R * 0.5)
    model.mesh.field.setNumber(2, "DistMax", R * 2)
    model.mesh.field.setAsBackgroundMesh(2)
    
    model.mesh.generate(2)
    
    # 2. Création de l'objet MeshData (selon ta doc)
    mesh_data = gmshio.model_to_mesh(model, comm, 0, gdim=2)
    
    # 3. Extraction explicite du maillage
    # Comme c'est un objet MeshData, on accède à l'attribut .mesh
    msh = mesh_data.mesh
    
    # 4. Nettoyage et retour
    gmsh.finalize()
    return msh

def solve_elasticity(msh, E, nu, F, comm):
    """Résout le problème d'élasticité sur le maillage donné."""
    dim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", 1, (dim,)))

    # --- 1) Physique ---
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(dim)

    # --- 2) Conditions aux limites ---
    # Encastrement à gauche (x=0)
    def left_boundary(x): return np.isclose(x[0], 0.0)
    f_left = mesh.locate_entities_boundary(msh, dim-1, left_boundary)
    bc_left = fem.dirichletbc(np.array([0, 0], dtype=ScalarType), 
                               fem.locate_dofs_topological(V, dim-1, f_left), V)

    # Traction à droite (x=L)
    x_coords = msh.geometry.x[:, 0]
    L_max = comm.allreduce(np.max(x_coords) if len(x_coords) > 0 else 0, op=MPI.MAX)
    def right_boundary(x): return np.isclose(x[0], L_max)
    f_right = mesh.locate_entities_boundary(msh, dim-1, right_boundary)
    tags = mesh.meshtags(msh, dim-1, f_right, np.full_like(f_right, 1))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    
    # --- 3) Formulation Variationnelle ---
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    # Force de traction F répartie sur le bord droit
    L_form = ufl.dot(ufl.as_vector([F, 0.0]), v) * ds(1)

    problem = LinearProblem(a, L_form, bcs=[bc_left], petsc_options_prefix="elasticity")
    uh = problem.solve()
    
    # --- 4) Calcul Von Mises ---
    s = sigma(uh) - (1./3)*ufl.tr(sigma(uh))*ufl.Identity(dim) # déviateur
    von_Mises = ufl.sqrt(3./2 * ufl.inner(s, s))
    
    W = fem.functionspace(msh, ("Lagrange", 1))
    p, w = ufl.TrialFunction(W), ufl.TestFunction(W)
    proj_prob = LinearProblem(ufl.inner(p, w) * ufl.dx, ufl.inner(von_Mises, w) * ufl.dx, petsc_options_prefix="vonmises")
    vm_field = proj_prob.solve()
    
    return uh, vm_field



def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size # Nombre de processeurs

    n_samples_total = 500  # Objectif total
    n_samples_local = n_samples_total // size # Part pour ce processeur
    
    # Initialisation du générateur aléatoire avec une graine différente par processeur
    rng = np.random.default_rng(42 + rank)
    
    local_data = []

    if rank == 0:
        print(f"Lancement de la génération de {n_samples_total} échantillons sur {size} cœurs...")

    for i in range(n_samples_local):
        # 1. Tirage des paramètres aléatoires (Bornes sécurisées)
        L = rng.uniform(1.5, 2.5)
        H = rng.uniform(0.8, 1.2)
        # On s'assure que le rayon R ne dépasse jamais 40% de la hauteur H
        R = rng.uniform(0.05, 0.4) * H 
        E = rng.uniform(70e9, 210e9)
        nu = rng.uniform(0.25, 0.35)
        F = rng.uniform(1e6, 1e8)

        try:
            # 2. Simulation
            msh = create_plate_with_hole_mesh(MPI.COMM_SELF, L, H, R)
            uh, vm_field = solve_elasticity(msh, E, nu, F, comm)

            # 3. Extraction des résultats
            u_max = np.max(np.linalg.norm(uh.x.array.reshape(-1, 2), axis=1))
            sig_max = np.max(vm_field.x.array)

            local_data.append([L, H, R, E, nu, F, u_max, sig_max])
            
            if rank == 0 and (i + 1) % 10 == 0:
                print(f"Progression : {((i + 1) * size / n_samples_total)*100:.1f}%")

        except Exception as e:
            print(f"Erreur sur le rang {rank} à l'itération {i}: {e}")
            continue

    # 4. Collecte des données sur le rang 0
    all_data = comm.gather(local_data, root=0)

    if rank == 0:
        # Aplatir la liste de listes
        flattened_data = [item for sublist in all_data for item in sublist]
        
        # Création du DataFrame et sauvegarde
        columns = ["L", "H", "R", "E", "nu", "F", "u_max", "sigma_max"]
        df = pd.DataFrame(flattened_data, columns=columns)
        df.to_csv("plate_with_hole_dataset.csv", index=False)
        print(f"\nDataset terminé ! Sauvegardé dans 'plate_with_hole_dataset.csv' ({len(df)} lignes)")

if __name__ == "__main__":
    main()
