import pandas as pd
import string
from collections import Counter

# Load data
df = pd.read_csv('SICK_train.txt', delimiter='\t')

# Split into subsets
neutral_data = df[df['entailment_judgment'] == 'NEUTRAL']
entailment_data = df[df['entailment_judgment'] == 'ENTAILMENT']
contradiction_data = df[df['entailment_judgment'] == 'CONTRADICTION']

# Basic text preprocessing
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Function to get top words for a specific column (sentence_A or B)
def get_top_words(data, column, top_n=10):
    tokens = data[column].dropna().apply(preprocess).sum()
    return Counter(tokens).most_common(top_n)

# Define which datasets and labels to use
datasets = {
    "NEUTRAL": neutral_data,
    "ENTAILMENT": entailment_data,
    "CONTRADICTION": contradiction_data
}

# Process and print top words for each label and sentence column
for label, data in datasets.items():
    print(f"\n--- {label} ---")
    
    top_a = get_top_words(data, 'sentence_A')
    print("Top words in sentence_A:")
    for word, count in top_a:
        print(f"{word}: {count}")
    
    top_b = get_top_words(data, 'sentence_B')
    print("\nTop words in sentence_B:")
    for word, count in top_b:
        print(f"{word}: {count}")
