import feedparser
import pandas as pd
from datetime import datetime

def fetch_arxiv_year(category, year, max_results=200):
    """
    Fetch papers from ArXiv for a given category and year.
    Returns a DataFrame with id, title, abstract, published.
    """
    base_url = "http://export.arxiv.org/api/query?"
    start = 0
    results_per_call = 100
    papers = []

    while start < max_results:
        query = f"search_query=cat:{category}+AND+submittedDate:[{year}01010000+TO+{year}12312359]"
        url = f"{base_url}{query}&start={start}&max_results={results_per_call}&sortBy=submittedDate&sortOrder=ascending"

        feed = feedparser.parse(url)
        if not feed.entries:
            break

        for entry in feed.entries:
            try:
                published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
                papers.append({
                    "id": entry.id,
                    "title": entry.title.strip().replace("\n", " "),
                    "abstract": entry.summary.strip().replace("\n", " "),
                    "published": published
                })
            except ValueError:
                print(f"Skipping entry with invalid date format: {entry.published}")
                continue
        
        start += results_per_call

    return pd.DataFrame(papers)