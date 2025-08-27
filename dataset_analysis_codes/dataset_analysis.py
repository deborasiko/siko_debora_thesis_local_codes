import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Install necessary libraries (already in user's code)
# !pip install pandas matplotlib numpy nltk seaborn sklearn gensim pyldavis wordcloud textblob spacy textstat

# Upload the file (already in user's code)
# uploaded = files.upload()

# Load the dataframe
df = pd.read_csv('SICK_train.txt', delimiter='\t')  # Adjust filename and delimiter if needed

# Print the column names to verify
print(df.columns)

# Print the first few rows to inspect the data
print(df.head())


neutral_data = df[df['entailment_judgment'] == 'NEUTRAL']

print(neutral_data.head())

entailment_data = df[df['entailment_judgment'] == 'ENTAILMENT']

print(entailment_data.head())

contradiction_data = df[df['entailment_judgment'] == 'CONTRADICTION']

print(contradiction_data.head())

# df['sentence_A'].str.len().hist()
# df['sentence_A'].apply(lambda x: len(str(x).split())).hist(bins=20)
# df['sentence_A'].str.split().map(lambda x: len(x)).hist()
# df['sentence_B'].str.split().map(lambda x: len(x)).hist()

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
df['sentence_A'].str.split().map(lambda x: len(x)).hist(ax=axs[0], color='skyblue') #length in words
# df['sentence_A'].str.len().hist(ax=axs[0], color='skyblue') #character length
#df['sentence_A'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist(ax=axs[0], color='skyblue') #average word(how many characters each word)length
axs[0].set_title('Histogram of word lengths in Premise (Sentence A)')
axs[0].set_xlabel('Number of Words')
axs[0].set_ylabel('Frequency')

# Plot histogram for sentence_B
df['sentence_B'].str.split().map(lambda x: len(x)).hist(ax=axs[1], color='salmon')
# df['sentence_B'].str.len().hist(ax=axs[1], color='salmon')
#df['sentence_B'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist(ax=axs[1], color='salmon')
axs[1].set_title('Histogram of word lengths in Hypothesis(Sentence B)')
axs[1].set_xlabel('Number of Words')
axs[1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()