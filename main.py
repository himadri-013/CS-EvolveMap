from data_fetch import fetch_arxiv_year
from preprocess import preprocess_text
from tfidf_manual import build_vocabulary, compute_tf, compute_idf
from clustering import dbscan_cosine
from cluster_semantics import extract_cluster_semantics, extract_cluster_keywords
from save_results import save_cluster_semantics, save_cluster_keywords
from semantic_linking import link_clusters_semantic
import json
import numpy as np

def run_pipeline(category, start_year, end_year, max_results=500):
    EPS = 0.8          # DBSCAN distance threshold (increased for better clustering)
    MIN_PTS = 10        # Min points for a robust cluster (increased)
    TOP_N = 10         # Keywords per cluster
    MAX_FEATURES = 3000 # Max features for the vocabulary

    year_docs = {}  # {year: [cleaned abstracts]}
    all_docs = []   # for global vocabulary

    # Pass 1: Fetch & preprocess
    for year in range(start_year, end_year + 1):
        print(f"\n📅 Fetching and cleaning year: {year}")
        df = fetch_arxiv_year(category, year, max_results=max_results)
        if df.empty:
            print(f"⚠ No data for {year}")
            year_docs[year] = []
            continue
        df["cleaned"] = df["abstract"].apply(preprocess_text)
        docs = df["cleaned"].tolist()
        year_docs[year] = docs
        all_docs.extend(docs)

    # Build global vocabulary & IDF
    print(f"\n📖 Building global vocabulary with top {MAX_FEATURES} features...")
    global_vocab = build_vocabulary(all_docs, max_features=MAX_FEATURES)
    global_idf = compute_idf(all_docs, global_vocab)

    # Pass 2: Process each year with fixed vocab
    for year in range(start_year, end_year + 1):
        print(f"\n⚙ Processing year: {year}")
        docs = year_docs[year]
        if not docs:
            print(f"⚠ Skipping {year}, no documents.")
            continue

        # Compute TF-IDF with fixed vocab
        tfidf_matrix = np.array([compute_tf(doc, global_vocab) * global_idf for doc in docs])

        # Cluster
        labels = dbscan_cosine(tfidf_matrix, eps=EPS, min_pts=MIN_PTS)
        num_clusters = len(set(labels) - {-1})
        print(f"Found {num_clusters} clusters for {year}")


        # Semantic vectors & keywords
        semantics = extract_cluster_semantics(tfidf_matrix, labels)
        keywords = extract_cluster_keywords(tfidf_matrix, global_vocab, labels, top_n=TOP_N)

        # Save
        save_cluster_semantics(year, semantics)
        save_cluster_keywords(year, keywords)
        if num_clusters > 0:
            print(f"✅ Saved {len(semantics)} clusters for {year}")

    # Linking
    print("\n🔗 Linking clusters semantically...")
    links = link_clusters_semantic(threshold=0.1) # Threshold can be adjusted
    print(f"✅ Found {len(links)} links between years")

    with open("results/topic_links_semantic.json", "w") as f:
        json.dump(links, f, indent=4)

    return links

if __name__ == "__main__":
    print("📊 Scientific Field Evolution Tracker (Semantic Version)\n")

    category = input("Enter ArXiv category code (e.g., cs.AI): ").strip()
    start_year = int(input("Enter start year (e.g., 2020): "))
    end_year = int(input("Enter end year (e.g., 2023): "))

    links = run_pipeline(category, start_year, end_year)

    print("\nSample semantic links across years:")
    for link in links[:10]:
        print(link)