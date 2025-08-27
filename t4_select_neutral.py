import pandas as pd

# Load the SICK dataset (assuming it's a CSV file)
# If it's tab-separated, you can use sep='\t', else adjust accordingly.
data = pd.read_csv('SICK_train.txt', sep='\t')  # Update the file path if needed

neutral_data = data[data['entailment_judgment'] == 'NEUTRAL']

print(neutral_data)

neutral_data.to_csv('neutral_sick_data.csv', index=False)
