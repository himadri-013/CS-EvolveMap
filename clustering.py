import numpy as np

def cosine_distance(a, b):
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return 1 - (dot / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 1.0

def region_query(data, idx, eps):
    neighbors = []
    for i in range(len(data)):
        if cosine_distance(data[idx], data[i]) <= eps:
            neighbors.append(i)
    return neighbors

def expand_cluster(data, labels, idx, neighbors, cluster_id, eps, min_pts):
    labels[idx] = cluster_id
    i = 0
    while i < len(neighbors):
        n_idx = neighbors[i]
        if labels[n_idx] == -1: # It was noise
            labels[n_idx] = cluster_id
        elif labels[n_idx] == 0: # It's unvisited
            labels[n_idx] = cluster_id
            new_neighbors = region_query(data, n_idx, eps)
            if len(new_neighbors) >= min_pts:
                # Add new neighbors to the list to be processed
                neighbors.extend(new_neighbors)
        i += 1

def dbscan_cosine(data, eps, min_pts):
    # labels: 0=unvisited, -1=noise, >0=cluster_id
    labels = [0] * len(data)
    cluster_id = 0
    for idx in range(len(data)):
        if labels[idx] != 0:
            continue
        neighbors = region_query(data, idx, eps)
        if len(neighbors) < min_pts:
            labels[idx] = -1 # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(data, labels, idx, neighbors, cluster_id, eps, min_pts)
    return labels