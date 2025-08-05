import numpy as np
import glob
import os
import json

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def link_clusters_semantic(start_year, end_year, threshold=0.1):
    """
    Loads cluster semantic vectors and links across a SPECIFIED range of years.
    The required arguments 'start_year' and 'end_year' come before the
    optional 'threshold' argument.
    """
    search_path = os.path.join("results", "clusters_semantics_*.npy")
    all_files = sorted(glob.glob(search_path))
    year_clusters = {}

    # Filter files to only include those within the selected date range
    files_in_range = []
    for year in range(start_year, end_year + 1):
        # Construct the expected filename for the current year
        expected_file = os.path.join("results", f"clusters_semantics_{year}.npy")
        if expected_file in all_files:
            files_in_range.append(expected_file)

    for file in files_in_range:
        try:
            year = int(file.split('_')[-1].split('.')[0])
            data = np.load(file, allow_pickle=True).item()
            if data:
                year_clusters[year] = data
        except (ValueError, IndexError):
            continue

    links = []
    years = sorted(year_clusters.keys())

    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i+1]

        if y1 not in year_clusters or y2 not in year_clusters:
            continue

        for c1, vec1 in year_clusters[y1].items():
            for c2, vec2 in year_clusters[y2].items():
                sim = cosine_similarity(vec1, vec2)
                if sim >= threshold:
                    links.append({
                        "source_year": y1,
                        "source_cluster": int(c1),
                        "target_year": y2,
                        "target_cluster": int(c2),
                        "similarity": float(sim)
                    })
    return links
