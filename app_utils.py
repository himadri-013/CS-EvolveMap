import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# --- Import your existing pipeline functions ---
from data_fetch import fetch_arxiv_year
from preprocess import preprocess_text
from tfidf_manual import build_vocabulary, compute_tf, compute_idf
from clustering import dbscan_cosine
from cluster_semantics import extract_cluster_semantics, extract_cluster_keywords
from save_results import save_cluster_semantics, save_cluster_keywords
from semantic_linking import link_clusters_semantic

# --- Utility and Helper Functions ---

def cleanup_json_files():
    results_dir = "results"
    if not os.path.exists(results_dir):
        return
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    for file_path in json_files:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

@st.cache_data
def load_keywords(year):
    path = os.path.join("results", f"clusters_keywords_{year}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def generate_cluster_title(keywords):
    if not keywords:
        return "Untitled Topic"
    title_words = [word.capitalize() for word in keywords[:3]]
    return " / ".join(title_words)

# --- Visualization Functions ---

def generate_pie_chart(cluster_sizes_for_year, year):
    """Creates a pie chart for a single year's topic distribution."""
    keywords_data = load_keywords(year)
    if not keywords_data or not cluster_sizes_for_year:
        return go.Figure().update_layout(title_text=f"No topic data for {year}")

    labels = []
    values = []
    for cluster_id, size in cluster_sizes_for_year.items():
        keywords = keywords_data.get(str(cluster_id), [])
        labels.append(generate_cluster_title(keywords))
        values.append(size)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', hole=.3)])
    fig.update_layout(title_text=f"Topic Distribution for {year}")
    return fig

def generate_sankey_diagram(links):
    if not links: return go.Figure()
    all_nodes, node_map = set(), {}
    for link in links:
        all_nodes.add((link['source_year'], str(link['source_cluster'])))
        all_nodes.add((link['target_year'], str(link['target_cluster'])))
    sorted_nodes = sorted(list(all_nodes), key=lambda x: (x[0], x[1]))
    node_map = {node: i for i, node in enumerate(sorted_nodes)}
    node_labels, node_hover_text = [], []
    for year, cluster_id in sorted_nodes:
        keywords = load_keywords(year).get(cluster_id, [])
        cluster_title = generate_cluster_title(keywords)
        label = f"<b>{year}</b><br>{cluster_title}"
        node_labels.append(label)
        hover_text = f"<b>{year} | Cluster {cluster_id}</b><br>Keywords: {', '.join(keywords)}"
        node_hover_text.append(hover_text)
    link_sources, link_targets, link_values, link_labels = [], [], [], []
    for link in links:
        source_node, target_node = (link['source_year'], str(link['source_cluster'])), (link['target_year'], str(link['target_cluster']))
        link_sources.append(node_map[source_node])
        link_targets.append(node_map[target_node])
        link_values.append(link['similarity'])
        link_labels.append(f"Similarity: {link['similarity']:.2f}")
    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(pad=25, thickness=20, line=dict(color="black", width=0.5), label=node_labels, customdata=node_hover_text, hovertemplate='%{customdata}<extra></extra>'),
        link=dict(source=link_sources, target=link_targets, value=link_values, label=link_labels, hovertemplate='Link from %{source.label} to %{target.label}<br>Similarity: %{label}<extra></extra>')
    ))
    fig.update_layout(title_text="Topic Flow Between Years", font_size=12, height=800)
    return fig

# --- Main analysis pipeline ---
def run_analysis_pipeline(category, start_year, end_year, status_placeholder):
    MAX_RESULTS = 500
    EPS = 0.8
    MIN_PTS = 3
    TOP_N = 10
    MAX_FEATURES = 3000
    year_docs, cluster_sizes_by_year = {}, {}
    all_docs = []
    for year in range(start_year, end_year + 1):
        status_placeholder.text(f"📅 Fetching and cleaning year: {year}...")
        df = fetch_arxiv_year(category, year, max_results=MAX_RESULTS)
        if df.empty:
            year_docs[year], cluster_sizes_by_year[year] = [], {}
            continue
        df["cleaned"] = df["abstract"].apply(preprocess_text)
        docs = df["cleaned"].tolist()
        year_docs[year] = docs
        all_docs.extend(docs)
    status_placeholder.text(f"📖 Building global vocabulary with top {MAX_FEATURES} features...")
    global_vocab = build_vocabulary(all_docs, max_features=MAX_FEATURES)
    global_idf = compute_idf(all_docs, global_vocab)
    for year in range(start_year, end_year + 1):
        status_placeholder.text(f"⚙️ Processing year: {year}...")
        docs = year_docs[year]
        if not docs:
            cluster_sizes_by_year[year] = {}
            continue
        tfidf_matrix = np.array([compute_tf(doc, global_vocab) * global_idf for doc in docs])
        labels = dbscan_cosine(tfidf_matrix, eps=EPS, min_pts=MIN_PTS)
        label_counts = Counter(labels)
        if -1 in label_counts: del label_counts[-1]
        cluster_sizes_by_year[year] = label_counts
        num_clusters = len(label_counts)
        status_placeholder.text(f"Found {num_clusters} clusters for {year}...")
        semantics = extract_cluster_semantics(tfidf_matrix, labels)
        keywords = extract_cluster_keywords(tfidf_matrix, global_vocab, labels, top_n=TOP_N)
        save_cluster_semantics(year, semantics)
        save_cluster_keywords(year, keywords)
    status_placeholder.text("🔗 Linking clusters across years...")
    links = link_clusters_semantic(start_year, end_year, threshold=0.1)
    with open("results/topic_links_semantic.json", "w") as f:
        json.dump(links, f, indent=4)
    status_placeholder.text("✅ Analysis complete!")
    return links, cluster_sizes_by_year