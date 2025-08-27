import pandas as pd
import matplotlib.pyplot as plt

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def jaccard_distance(set1, set2):
    return 1 - jaccard_similarity(set1, set2)

def tokenize_to_set(premise, hypothesis):
    tokens_p = set(premise.lower().split())
    tokens_h = set(hypothesis.lower().split())
    return tokens_p, tokens_h


def calc_jaccard_every_pair(data):
    jaccard_similarities = []
    jaccard_distances = []
    for index, row in data.iterrows():
            premise = row['sentence_A']
            hypothesis = row['sentence_B']
            tokens_p, tokens_h = tokenize_to_set(premise,hypothesis)
            sim = jaccard_similarity(tokens_p,tokens_h)
            dist = jaccard_distance(tokens_p, tokens_h)
            jaccard_distances.append(dist)
            jaccard_similarities.append(sim)
    return jaccard_similarities, jaccard_distances


df = pd.read_csv('SICK_train.txt', delimiter='\t')
neutral_data = df[df['entailment_judgment'] == 'NEUTRAL']
entailment_data = df[df['entailment_judgment'] == 'ENTAILMENT']
contradiction_data = df[df['entailment_judgment'] == 'CONTRADICTION']

jaccard_sim, jaccard_dist = calc_jaccard_every_pair(df)

# Compute Jaccard similarities
jaccard_neutral, jdist_neut = calc_jaccard_every_pair(neutral_data)
jaccard_entailment, jdist_entail = calc_jaccard_every_pair(entailment_data)
jaccard_contradiction, jdist_contr = calc_jaccard_every_pair(contradiction_data)

# Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(jaccard_neutral, bins=30, alpha=0.5, label='Neutral', color='skyblue')
plt.hist(jaccard_entailment, bins=30, alpha=0.5, label='Entailment', color='lightgreen')
plt.hist(jaccard_contradiction, bins=30, alpha=0.5, label='Contradiction', color='salmon')

plt.title('Jaccard Similarity Distribution by Entailment Type')
plt.xlabel('Jaccard Similarity')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
