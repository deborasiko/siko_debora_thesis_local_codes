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

def getDifferences(data):
    differences = []
    for index, row in data.iterrows():
            premise = row['sentence_A']
            hypothesis = row['sentence_B']
            plen = len(premise.split())
            hlen = len(hypothesis.split())
            difference = plen - hlen
            differences.append(difference)
    premiseWordLength = data['sentence_A'].str.split().map(lambda x: len(x)) #length in words
    hypothesisWordLength = data['sentence_B'].str.split().map(lambda x: len(x)) #length in words
    return premiseWordLength, hypothesisWordLength, differences

premiseLength, hypothesisLength, diffs = getDifferences(neutral_data)
neutPL, neutHL, neutdiffs = getDifferences(neutral_data)
entailPL, entailHL, entaildiffs = getDifferences(entailment_data)
contrPL, contrHL, contrdiffs = getDifferences(contradiction_data)

# Plot histograms for sentence lengths (in characters)
# fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# axs[0].hist(premiseLength, alpha=0.5, color='blue')
# axs[0].hist(hypothesisLength, alpha=0.5, color = 'salmon')
# axs[0].set_title('Histogram of word lengths in Premise and Hypothesis Compared')
# axs[0].set_xlabel('Number of Words')
# axs[0].set_ylabel('Frequency')
# axs[0].set_xticks(range(int(min(premiseLength.min(), hypothesisLength.min())), int(max(premiseLength.max(), hypothesisLength.max())) + 1))

# axs[1].hist(diffs)
# axs[1].set_title('Histogram of Word Count Differences')
# axs[1].set_xlabel('Premise Length - Hypothesis Length')
# axs[1].set_ylabel('Frequency')
# axs[1].set_xticks(range(min(diffs), max(diffs) + 1))

# axs[2].hist(neutdiffs, alpha=0.5, label='Neutral', color='skyblue')
# axs[2].hist(entaildiffs, alpha=0.5, label='Entailment', color='lightgreen')
# axs[2].hist(contrdiffs, alpha=0.5, label='Contradiction', color='salmon')
# axs[2].set_title('Histogram of Word Count Differences')
# axs[2].set_xlabel('Number of Words')
# axs[2].set_ylabel('Frequency')
# axs[2].set_xticks(range(min(min(neutdiffs),min(entaildiffs),min(contrdiffs)), max(max(neutdiffs),max(entaildiffs),max(contrdiffs)) + 1))
# axs[2].legend(loc='upper right')

plt.hist(neutdiffs, alpha=0.5, label='Neutral', color='skyblue')
plt.hist(entaildiffs, alpha=0.5, label='Entailment', color='lightgreen')
plt.hist(contrdiffs, alpha=0.5, label='Contradiction', color='salmon')
# plt.set_title('Histogram of Word Count Differences')
# plt.set_xlabel('Number of Words')
# plt.set_ylabel('Frequency')
# plt.set_xticks(range(min(min(neutdiffs),min(entaildiffs),min(contrdiffs)), max(max(neutdiffs),max(entaildiffs),max(contrdiffs)) + 1))
# plt.legend(loc='upper right')

plt.title('Histogram of Word Count Differences')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.xticks(range(min(min(neutdiffs),min(entaildiffs),min(contrdiffs)), max(max(neutdiffs),max(entaildiffs),max(contrdiffs)) + 1))
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()

# plt.tight_layout(pad=3.0)
# plt.show()