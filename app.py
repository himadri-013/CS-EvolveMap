import streamlit as st
from datetime import datetime
import app_utils as utils

# --- Run the cleanup function at the start of every session ---
utils.cleanup_json_files()

# --- Initialize session state keys ---
# This ensures that the keys exist on the first run
if 'analysis_complete' not in st.session_state:
    st.session_state['analysis_complete'] = False
if 'results_links' not in st.session_state:
    st.session_state['results_links'] = []
if 'results_cluster_sizes' not in st.session_state:
    st.session_state['results_cluster_sizes'] = {}


# --- Mapping of user-friendly names to ArXiv category codes ---
CS_CATEGORIES = {
    "Artificial Intelligence": "cs.AI",
    "Machine Learning": "cs.LG",
    "Computer Vision and Pattern Recognition": "cs.CV",
    "Computation and Language": "cs.CL",
    "Robotics": "cs.RO",
    "Human-Computer Interaction": "cs.HC",
    "Networking and Internet Architecture": "cs.NI",
    "Cryptography and Security": "cs.CR",
    "Databases": "cs.DB",
    "Software Engineering": "cs.SE",
    "Data Structures and Algorithms": "cs.DS",
    "Graphics": "cs.GR",
    "Operating Systems": "cs.OS",
    "Distributed, Parallel, and Cluster Computing": "cs.DC"
}

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("🔬 Scientific Field Evolution Tracker in Computer Science Research Topics")

with st.sidebar:
    st.header("Analysis Configuration")
    
    with st.form("input_form"):
        selected_field = st.selectbox(
            "Choose a Computer Science field:",
            options=list(CS_CATEGORIES.keys())
        )
        current_year = datetime.now().year
        year_range = st.slider(
            "Select year range:",
            min_value=2010,
            max_value=current_year,
            value=(current_year - 4, current_year - 1)
        )
        submitted = st.form_submit_button("Run Analysis")

# --- Logic for Running the Analysis ---
if submitted:
    category_code = CS_CATEGORIES[selected_field]
    start_year, end_year = year_range
    
    if start_year >= end_year:
        st.error("Error: Start year must be before end year.")
    else:
        st.info(f"Running analysis for category '{selected_field}' ({category_code}) from {start_year} to {end_year}...")
        status_placeholder = st.empty()
        
        with st.spinner("Analysis in progress... This may take several minutes."):
            try:
                # Run the pipeline and save results to session state
                links, cluster_sizes = utils.run_analysis_pipeline(category_code, start_year, end_year, status_placeholder)
                st.session_state['results_links'] = links
                st.session_state['results_cluster_sizes'] = cluster_sizes
                st.session_state['analysis_complete'] = True

            except Exception as e:
                st.error(f"An error occurred during the analysis: {e}")
                st.exception(e) # This will print the full traceback for debugging
                st.session_state['analysis_complete'] = False

# --- Logic for Displaying Results (runs if analysis is complete) ---
if st.session_state['analysis_complete']:
    
    # Retrieve results from session state
    links = st.session_state['results_links']
    cluster_sizes = st.session_state['results_cluster_sizes']
    
    if not any(cluster_sizes.values()):
        st.warning("No topics were found in the selected date range.")
    else:
        st.success("Analysis complete!")
        
        # --- Pie Chart Section ---
        st.header("Topic Distribution by Year")
        valid_years = [year for year, sizes in cluster_sizes.items() if sizes]
        if valid_years:
            # The selectbox can now be interacted with without losing state
            selected_year = st.selectbox("Choose a year to inspect:", options=valid_years)
            pie_fig = utils.generate_pie_chart(cluster_sizes[selected_year], selected_year)
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No topics available to display in a pie chart.")

        # --- Sankey Diagram for flows ---
        st.header("Topic Flow Between Years")
        if not links:
            st.warning("No semantic links were found between the discovered topics.")
        else:
            sankey_fig = utils.generate_sankey_diagram(links)
            st.plotly_chart(sankey_fig, use_container_width=True)

        # --- Expander for detailed text results ---
        with st.expander("Show Detailed Links List"):
            if not links:
                st.write("No links to display.")
            else:
                sorted_links = sorted(links, key=lambda k: (k['source_year'], k['similarity']), reverse=True)
                for link in sorted_links:
                    source_year, target_year = link['source_year'], link['target_year']
                    source_cluster_id, target_cluster_id = str(link['source_cluster']), str(link['target_cluster'])
                    source_keywords_data, target_keywords_data = utils.load_keywords(source_year), utils.load_keywords(target_year)
                    source_keywords_list = source_keywords_data.get(source_cluster_id, [])
                    target_keywords_list = target_keywords_data.get(target_cluster_id, [])
                    source_title = utils.generate_cluster_title(source_keywords_list)
                    target_title = utils.generate_cluster_title(target_keywords_list)
                    source_keywords_str = ", ".join(f"`{k}`" for k in source_keywords_list)
                    target_keywords_str = ", ".join(f"`{k}`" for k in target_keywords_list)
                    st.markdown(f"#### {source_year} ➔ {target_year} (Similarity: {link['similarity']:.3f})")
                    st.markdown(f"- **From Topic in {source_year}:** *{source_title}*\n  - **Keywords:** {source_keywords_str}")
                    st.markdown(f"- **To Topic in {target_year}:** *{target_title}*\n  - **Keywords:** {target_keywords_str}")
                    st.divider()