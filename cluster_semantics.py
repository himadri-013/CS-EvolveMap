import numpy as np

def extract_cluster_semantics(tfidf_matrix, labels):
    """
    Returns dict { cluster_id: semantic_vector }
    where semantic_vector is the mean TF-IDF vector for that cluster.
    """
    cluster_semantics = {}
    unique_labels = set(labels)
    for cluster_id in unique_labels:
        if cluster_id == -1: # -1 is for noise points
            continue
        # Get indices of documents in the current cluster
        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        if not cluster_indices:
            continue
        
        cluster_docs_vectors = tfidf_matrix[cluster_indices]
        avg_vector = np.mean(cluster_docs_vectors, axis=0)
        cluster_semantics[cluster_id] = avg_vector
        
    return cluster_semantics

def extract_cluster_keywords(tfidf_matrix, vocab, labels, top_n=5):
    """
    Returns dict { cluster_id: [top keywords] } for labeling clusters later.
    """
    cluster_keywords = {}
    unique_labels = set(labels)
    for cluster_id in unique_labels:
        if cluster_id == -1: # -1 is for noise points
            continue
            
        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        if not cluster_indices:
            continue
        
        cluster_docs_vectors = tfidf_matrix[cluster_indices]
        avg_tfidf = np.mean(cluster_docs_vectors, axis=0)
        
        # Get indices of top N tf-idf scores
        top_indices = np.argsort(avg_tfidf)[-top_n:][::-1]
        cluster_keywords[cluster_id] = [vocab[i] for i in top_indices]
        
    return cluster_keywords
