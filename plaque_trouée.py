import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
import dolfinx.io.gmsh as gmshio
import ufl
from petsc4py.PETSc import ScalarType
import pandas as pd
import os

# --- 1. FONCTIONS DE MAILLAGE ET SOLVEUR (Inchangées ou légèrement adaptées) ---

def create_plate_with_hole_mesh(comm, L, H, R):
    """Crée un maillage de plaque trouée."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model
    model.add("Plate_with_Hole")
    
    lc = H / 12  # Maillage un peu plus grossier pour limiter la taille du CSV
    
    rect = model.occ.addRectangle(0, 0, 0, L, H)
    circle = model.occ.addDisk(L/2, H/2, 0, R, R)
    out, _ = model.occ.cut([(2, rect)], [(2, circle)])
    
    model.occ.synchronize()
    surface_tag = out[0][1]
    model.addPhysicalGroup(2, [surface_tag], tag=1)
    model.setPhysicalName(2, 1, "Plate_Surface")
    
    # Raffinement
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
    
    # On retourne le modèle complet pour extraire la topologie si besoin
    mesh_data = gmshio.model_to_mesh(model, comm, 0, gdim=2)
    msh = mesh_data.mesh
    gmsh.finalize()
    return msh

def solve_elasticity(msh, E, nu, F):
    """Résout le problème et retourne l'espace de fonction V pour mapper les nœuds."""
    dim = msh.geometry.dim
    # Degré 1 (Linéaire) pour que les DOFs correspondent aux sommets du maillage
    V = fem.functionspace(msh, ("Lagrange", 1, (dim,)))

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    def epsilon(u): return ufl.sym(ufl.grad(u))
    def sigma(u): return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(dim)

    # BCs
    def left_boundary(x): return np.isclose(x[0], 0.0)
    f_left = mesh.locate_entities_boundary(msh, dim-1, left_boundary)
    bc_left = fem.dirichletbc(np.array([0, 0], dtype=ScalarType), 
                            fem.locate_dofs_topological(V, dim-1, f_left), V)

    # Load
    def right_boundary(x): 
        # On utilise une tolérance large car le maillage n'est pas structuré
        return np.isclose(x[0], np.max(msh.geometry.x[:, 0]))
        
    f_right = mesh.locate_entities_boundary(msh, dim-1, right_boundary)
    tags = mesh.meshtags(msh, dim-1, f_right, np.full_like(f_right, 1))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L_form = ufl.dot(ufl.as_vector([F, 0.0]), v) * ds(1)

    problem = LinearProblem(a, L_form, bcs=[bc_left], petsc_options_prefix="elasticity")
    uh = problem.solve()
    
    return uh, V  # On retourne V pour accéder aux coordonnées des nœuds

# --- 2. FONCTION D'EXTRACTION DES DONNÉES ---

def extract_simulation_data(sim_id, msh, V, uh, E, nu, F):
    """Extrait les données nodales et la connectivité pour le GNN."""
    
    # A. Extraction des Nœuds (Features + Labels)
    # ---------------------------------------------
    # Coordonnées des DOFs (Degrees of Freedom)
    # Pour Lagrange 1, il y a un mapping direct avec les sommets
    dof_coords = V.tabulate_dof_coordinates()
    u_values = uh.x.array.reshape(-1, 2) # [ux, uy] pour chaque nœud
    
    # On récupère x et y
    x_nodes = dof_coords[:, 0]
    y_nodes = dof_coords[:, 1]
    
    num_nodes = len(x_nodes)
    
    # Création des vecteurs de features
    # Boundary Conditions : Force à droite
    load_x = np.zeros(num_nodes)
    max_x = np.max(x_nodes)
    # Les nœuds sur le bord droit reçoivent la charge F (simplifié)
    right_mask = np.isclose(x_nodes, max_x, atol=1e-3)
    load_x[right_mask] = F
    
    load_y = np.zeros(num_nodes) # Pas de force verticale dans ce cas
    
    # Boundary Conditions : Fixe à gauche
    is_fixed = np.isclose(x_nodes, 0.0, atol=1e-3).astype(int)
    
    # Construction de la liste des dictionnaires (1 ligne par nœud)
    node_rows = []
    for i in range(num_nodes):
        node_rows.append({
            "SimulationID": sim_id,
            "NodeID": i,
            "x": x_nodes[i],
            "y": y_nodes[i],
            "E": E,
            "nu": nu,
            "Fx": load_x[i],
            "Fy": load_y[i],
            "isFixed": is_fixed[i],
            "ux": u_values[i, 0],
            "uy": u_values[i, 1]
        })
        
    # B. Extraction de la Connectivité (Edges)
    # ----------------------------------------
    # msh.topology.connectivity(dim, dim) donne les voisins
    # Mais pour le GNN, on veut souvent les arêtes du maillage.
    # On va extraire les cellules (triangles) pour reconstruire le graphe plus tard
    # msh.topology.index_map(2) donne les cellules
    
    # Astuce : Pour un GNN sur maillage non-structuré, on a besoin de savoir
    # quels nœuds forment un triangle.
    # V.dofmap.list donne la liste des DOFs pour chaque cellule
    dofmap = V.dofmap.list.array.reshape(-1, 3) # Triangles (3 nœuds)
    
    connectivity_rows = []
    for cell_idx, cell_dofs in enumerate(dofmap):
        connectivity_rows.append({
            "SimulationID": sim_id,
            "n1": cell_dofs[0],
            "n2": cell_dofs[1],
            "n3": cell_dofs[2]
        })

    return node_rows, connectivity_rows

# --- 3. MAIN ---

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    n_samples_total = 20 # Exemple réduit pour tester, mettez 500 pour le vrai run
    n_samples_local = n_samples_total // size
    
    rng = np.random.default_rng(42 + rank)
    
    local_nodes = []
    local_connectivity = []

    if rank == 0:
        print(f"Génération de {n_samples_total} simulations (Nodes + Edges)...")

    for i in range(n_samples_local):
        # ID unique global pour la simulation
        sim_id = rank * n_samples_local + i
        
        # Paramètres
        L = rng.uniform(1.5, 2.5)
        H = rng.uniform(0.8, 1.2)
        R = rng.uniform(0.1, 0.3) * H
        E = rng.uniform(10e9, 210e9) # De 10 à 210 GPa
        nu = rng.uniform(0.2, 0.4)
        F = rng.uniform(1e5, 1e7)    # Attention aux unités

        try:
            # Simulation Locale (MPI.COMM_SELF est crucial ici)
            msh = create_plate_with_hole_mesh(MPI.COMM_SELF, L, H, R)
            uh, V = solve_elasticity(msh, E, nu, F)
            
            # Extraction
            nodes, topology = extract_simulation_data(sim_id, msh, V, uh, E, nu, F)
            local_nodes.extend(nodes)
            local_connectivity.extend(topology)
            
            if rank == 0:
                print(f"Sim {sim_id} terminée ({len(nodes)} nœuds).")

        except Exception as e:
            print(f"Erreur Sim {sim_id}: {e}")
            continue

    # Collecte MPI
    all_nodes = comm.gather(local_nodes, root=0)
    all_topo = comm.gather(local_connectivity, root=0)

    if rank == 0:
        # Aplatir et Sauvegarder
        flat_nodes = [item for sublist in all_nodes for item in sublist]
        flat_topo = [item for sublist in all_topo for item in sublist]
        
        df_nodes = pd.DataFrame(flat_nodes)
        df_topo = pd.DataFrame(flat_topo)
        
        # On sauvegarde 2 fichiers : Les données aux nœuds ET la structure
        df_nodes.to_csv("db.csv", index=False)
        df_topo.to_csv("connectivity.csv", index=False)
        
        print("\nTerminé !")
        print(f"-> 'db.csv' : {len(df_nodes)} lignes (Features)")
        print(f"-> 'connectivity.csv' : {len(df_topo)} lignes (Triangles)")

if __name__ == "__main__":
    main()