import re
import pandas as pd

# A slightly larger set of stopwords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "don",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "t", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
    # Domain-specific words that might be noise
    "paper", "results", "study", "show", "based", "propose", "present", "model",
    "models", "approach", "method", "methods", "algorithm", "algorithms"
}


def preprocess_text(text):
    """
    Cleans and preprocesses a text:
    - Lowercase
    - Remove punctuation/numbers
    - Remove stopwords
    - Normalize multiple spaces
    """
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[\(\)\[\]\{\}]", " ", text)  # remove brackets
    text = re.sub(r"[^a-z]", " ", text)          # keep only letters
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    cleaned = " ".join(words)
    return re.sub(r"\s+", " ", cleaned).strip()  # normalize spaces
