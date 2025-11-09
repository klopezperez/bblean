import argparse
import pickle

import numpy as np

import bblean
import bblean.similarity as iSIM

# Parse command line arguments
parser = argparse.ArgumentParser(description="Cluster a large set of molecules using BitBirch.")
parser.add_argument(
    "-s",
    "--smiles",
    type=str,
    required=True,
    help="Path to input SMILES file.",
)
args = parser.parse_args()

# Load SMILES and create fingerprints
# Please ensure that you do not have any invalid SMILES in your input file
print("Loading SMILES and computing fingerprints...")
smiles = bblean.load_smiles(args.smiles)
fps, invalid_smiles = bblean.fps_from_smiles(smiles, 
                                             pack=True, 
                                             skip_invalid=True, 
                                             n_features=2048, 
                                             kind="ecfp4")
smiles = np.delete(smiles, invalid_smiles, axis=0)
assert len(smiles) == len(fps), "Number of SMILES and fingerprints do not match!"


# Select the optimal threshold
if len(fps) > 10_000_000:
    random_sample = np.random.choice(len(fps), size=1_000_000, replace=False)
    fps_sample = fps[random_sample]
    representative_samples = iSIM.jt_stratified_sampling(fps_sample, n_samples=50)
    representative_samples = random_sample[representative_samples]
    del fps_sample
else:
    representative_samples = iSIM.jt_stratified_sampling(fps, n_samples=50)    

sim_matrix = iSIM.jt_sim_matrix_packed(fps[representative_samples])
sim_matrix = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
average_sim = np.mean(sim_matrix)
std = np.std(sim_matrix)
del sim_matrix

optimal_threshold = average_sim + 3.5 * std

# Do the initial clustering
bb_tree = bblean.BitBirch(branching_factor=50, threshold=optimal_threshold, merge_criterion="diameter")
bb_tree.fit(fps)

# Refine to obtain better clusters
bb_tree.recluster_inplace(iterations=5, extra_threshold=std, shuffle=False, verbose=True)

# Obtain final output
clusters = bb_tree.get_cluster_mol_ids()

# Obtain the medoids
cluster_size = []
fingerprints_medoids = []
smiles_medoids = []
for cluster in clusters:
    cluster_size.append(len(cluster))
    fps_cluster = fps[cluster]
    medoid_id, medoid_fp = iSIM.jt_isim_medoid(
        fps_cluster, input_is_packed=True, n_features=2048, pack=True
    )
    fingerprints_medoids.append(bblean.unpack_fingerprints(medoid_fp))
    smiles_medoids.append(smiles[cluster[medoid_id]])

# Save the medoids to a file
output_name = args.smiles.split(".")[0] + "_medoids.pkl"
with open("chembl_medoids.pkl", "wb") as f:
    pickle.dump(
        {
            "smiles": smiles_medoids,
            "fingerprints": fingerprints_medoids,
            "cluster_size": cluster_size,
        },
        f,
    )




