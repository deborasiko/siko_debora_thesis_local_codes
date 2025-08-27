import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataframe
df = pd.read_csv('SICK_train.txt', delimiter='\t')

# Check data
print(df.columns)
print(df.head())

# Subsets
neutral_data = df[df['entailment_judgment'] == 'NEUTRAL']
entailment_data = df[df['entailment_judgment'] == 'ENTAILMENT']
contradiction_data = df[df['entailment_judgment'] == 'CONTRADICTION']

# Plot histograms for sentence lengths (in characters)
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Sentence A histogram
axs[0].hist(neutral_data['sentence_A'].str.len(), bins=30, alpha=0.5, label='Neutral', color='skyblue')
axs[0].hist(entailment_data['sentence_A'].str.len(), bins=30, alpha=0.5, label='Entailment', color='lightgreen')
axs[0].hist(contradiction_data['sentence_A'].str.len(), bins=30, alpha=0.5, label='Contradiction', color='salmon')
axs[0].set_title('Histogram of Sentence A (Premise) Lengths by Entailment Type')
axs[0].set_xlabel('Number of Characters')
axs[0].set_ylabel('Frequency')
axs[0].legend()

# Sentence B histogram
axs[1].hist(neutral_data['sentence_B'].str.len(), bins=30, alpha=0.5, label='Neutral', color='skyblue')
axs[1].hist(entailment_data['sentence_B'].str.len(), bins=30, alpha=0.5, label='Entailment', color='lightgreen')
axs[1].hist(contradiction_data['sentence_B'].str.len(), bins=30, alpha=0.5, label='Contradiction', color='salmon')
axs[1].set_title('Histogram of Sentence B (Hypothesis) Lengths by Entailment Type')
axs[1].set_xlabel('Number of Characters')
axs[1].set_ylabel('Frequency')
axs[1].legend()

plt.tight_layout()
plt.show()
