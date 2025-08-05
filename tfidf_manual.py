import numpy as np
from collections import Counter

def build_vocabulary(docs, max_features=2000):
    """
    Builds a vocabulary from the documents, limited to the most frequent words.
    """
    word_counts = Counter(" ".join(docs).split())
    most_common_words = [word for word, count in word_counts.most_common(max_features)]
    return sorted(most_common_words)

def compute_tf(doc, vocab):
    words = doc.split()
    word_count = len(words)
    return np.array([words.count(term) / word_count if word_count else 0 for term in vocab])

def compute_idf(docs, vocab):
    N = len(docs)
    # Create a set of words for each document for faster 'in' check
    docs_sets = [set(doc.split()) for doc in docs]
    return np.array([
        np.log((N + 1) / (sum(1 for doc_set in docs_sets if term in doc_set) + 1)) + 1
        for term in vocab
    ])

def compute_tfidf_matrix(docs):
    vocab = build_vocabulary(docs)
    idf = compute_idf(docs, vocab)
    tfidf = np.array([compute_tf(doc, vocab) * idf for doc in docs])
    return tfidf, vocab