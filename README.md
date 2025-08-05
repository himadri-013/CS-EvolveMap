# CS EvolveMap

**CS EvolveMap** is a Streamlit-based interactive tool that analyzes and visualizes the evolution of research topics in Computer Science using ArXiv data.

---

## 📌 Overview

CS EvolveMap allows users to explore how key research topics in various CS fields have changed over the years. It does this by clustering research abstracts from ArXiv and then linking these clusters over time based on semantic similarity.

Try the app here : https://cs-evolvemap-khetxmjnpf8xikenzfs7gs.streamlit.app/
---

## ⚙️ How It Works

### 1. **User Input (via Streamlit UI)**
- The user selects a CS field (e.g., Machine Learning, AI, NLP).
- Specifies a year range for analysis.
- Clicks "Run Analysis" to begin processing.

### 2. **Data Fetching**
- Module: `data_fetch.py` *(not uploaded)*
- Uses ArXiv API to fetch abstracts for the selected category and years.

### 3. **Text Preprocessing**
- Module: `preprocess.py`
- Cleans the abstracts (lowercasing, removing stopwords, punctuation, etc.) to prepare for vectorization.

### 4. **TF-IDF Vectorization**
- Module: `tfidf_manual.py` *(not uploaded)*
- Builds a global vocabulary from all abstracts and computes TF-IDF vectors per document.

### 5. **Clustering**
- Module: `clustering.py` *(not uploaded)*
- Applies DBSCAN with cosine similarity to group related papers into topic clusters.

### 6. **Cluster Semantics Extraction**
- Module: `cluster_semantics.py`
- Computes average TF-IDF vectors for each cluster.
- Extracts top keywords to represent the cluster.

### 7. **Saving Results**
- Module: `save_results.py` *(not uploaded)*
- Stores cluster semantics and keywords for later use.

### 8. **Semantic Linking Across Years**
- Module: `semantic_linking.py`
- Computes cosine similarity between clusters in consecutive years.
- Links clusters if similarity crosses a defined threshold.

### 9. **Visualization and UI Rendering**
- Module: `app_utils.py`
  - Generates pie charts for topic distribution per year.
  - Builds Sankey diagrams for topic transitions across years.
  - Loads cluster data and formats keyword-based summaries.

- Module: `app.py`
  - Main Streamlit interface connecting all modules.
  - Handles user interaction, analysis execution, and result display.

---

## 📊 Visual Output

- **Pie Charts:** Show topic proportions per year.
- **Sankey Diagram:** Illustrates semantic flow of research topics across years.
- **Detailed Keyword Views:** Expanders show key terms and topic linkage details.

---

## 📁 File Structure

├── app.py # Main Streamlit frontend
├── app_utils.py # UI logic, visualizations, pipeline runner
├── preprocess.py # Text cleaning and stopword removal
├── cluster_semantics.py # Cluster vector computation and keyword extraction
├── semantic_linking.py # Inter-year topic linking via cosine similarity
├── requirements.txt # Project dependencies
