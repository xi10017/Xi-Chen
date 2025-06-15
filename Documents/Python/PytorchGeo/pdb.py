import os
from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def extract_protein_atoms(structure):
    protein_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':
                    atoms = list(residue.get_atoms())
                    coords = [atom.coord for atom in atoms]
                    protein_atoms.extend(coords)
    return np.array(protein_atoms)

def extract_ligand_atoms_from_sdf(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError("Could not read ligand from SDF file.")
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    return np.array(coords)

def compute_gaussian_kernel(distance_matrix, sigma=1.0):
    return np.exp(-distance_matrix**2 / (2 * sigma**2))

def compute_bipartite_laplacian(kernel):
    n_protein, n_ligand = kernel.shape
    zero_protein = np.zeros((n_protein, n_protein))
    zero_ligand = np.zeros((n_ligand, n_ligand))
    A = np.block([
        [zero_protein, kernel],
        [kernel.T, zero_ligand]
    ])
    D = np.diag(A.sum(axis=1))
    L = D - A
    return L

def laplacian_summary_stats(laplacian):
    # Extract largest connected component
    laplacian_sparse = csr_matrix(laplacian)
    adjacency = -laplacian_sparse.copy()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[adjacency.data < 0] *= -1
    n_components, labels = connected_components(adjacency, directed=False)
    largest = np.argmax(np.bincount(labels))
    mask = (labels == largest)
    laplacian_lcc = laplacian_sparse[mask][:, mask]
    laplacian_lcc_dense = laplacian_lcc.toarray()
    vals = np.linalg.eigvalsh(laplacian_lcc_dense)
    # Remove near-zero eigenvalues
    nonzero_vals = vals[np.abs(vals) > 1e-8]
    # Summary statistics
    stats = [
        np.sum(nonzero_vals),
        np.mean(nonzero_vals),
        np.max(nonzero_vals),
        np.min(nonzero_vals),
        np.std(nonzero_vals)
    ]
    return stats

def process_complex(pdb_path, sdf_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    protein_coords = extract_protein_atoms(structure)
    ligand_coords = extract_ligand_atoms_from_sdf(sdf_path)
    if len(protein_coords) == 0 or len(ligand_coords) == 0:
        raise ValueError("Could not extract sufficient protein or ligand atoms.")
    dist_matrix = cdist(protein_coords, ligand_coords, metric='euclidean')
    kernel = compute_gaussian_kernel(dist_matrix, sigma=2.0)
    mask = dist_matrix <= 12.0
    kernel[~mask] = 0.0
    protein_mask = kernel.sum(axis=1) > 0
    ligand_mask = kernel.sum(axis=0) > 0
    kernel_reduced = kernel[np.ix_(protein_mask, ligand_mask)]
    laplacian = compute_bipartite_laplacian(kernel_reduced)
    return laplacian_summary_stats(laplacian)

# --- MAIN PIPELINE ---

# Example: Load CASF-2007 data
# You need to provide a list of (pdb_path, sdf_path, binding_affinity) for each complex
# For demonstration, let's assume you have a CSV file with these columns

import pandas as pd

casf_df = pd.read_csv("casf2007_index.csv")  # columns: pdb_path,sdf_path,affinity

features = []
targets = []

for idx, row in casf_df.iterrows():
    try:
        stats = process_complex(row['pdb_path'], row['sdf_path'])
        features.append(stats)
        targets.append(row['affinity'])
        print(f"Processed {row['pdb_path']}")
    except Exception as e:
        print(f"Failed {row['pdb_path']}: {e}")

X = np.array(features)
y = np.array(targets)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting Regression
reg = GradientBoostingRegressor(random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Test R2:", r2_score(y_test, y_pred))