import pandas as pd

# Load the SICK dataset
data = pd.read_csv('SICK_train.txt', sep='\t')  # Adjust if your file uses a different separator

# Filter for entailment judgments
entailment_data = data[data['entailment_judgment'] == 'ENTAILMENT']

print(entailment_data)

# Save to a .txt file using comma separation
entailment_data.to_csv('entailment_sick_data.txt', sep=',', index=False)
