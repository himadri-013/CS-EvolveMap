import os
import numpy as np
import json

def save_cluster_semantics(year, cluster_semantics):
    os.makedirs("results", exist_ok=True)  # ensure folder exists
    # Save as a dictionary
    np.save(f"results/clusters_semantics_{year}.npy", cluster_semantics)

def save_cluster_keywords(year, cluster_keywords):
    os.makedirs("results", exist_ok=True)  # ensure folder exists
    # Convert numpy int keys to python int if necessary
    keywords_to_save = {int(k): v for k, v in cluster_keywords.items()}
    with open(f"results/clusters_keywords_{year}.json", "w") as f:
        json.dump(keywords_to_save, f, indent=4)

