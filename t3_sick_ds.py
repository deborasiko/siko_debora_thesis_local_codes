import requests
import pandas as pd

# Function to make an API call to LM Studio
def get_nli_result(premise, hypothesis):
    url = "http://192.168.100.44:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Define the NLI prompt
    data = {
        "model": "mistral-7b-instruct-v0.3",  # Make sure this matches your model name
        "messages": [
            {"role": "user", "content": f"Premise: {premise}\nHypothesis: {hypothesis}\nWhat is the relationship?"}
        ],
        "temperature": 0.2
    }

    # Send the request
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        response_json = response.json()

        # Check if the 'choices' key exists
        if "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        else:
            print("Error: 'choices' key not found")
            return None
    else:
        print(f"Error: {response.status_code}")
        return None

# Function to compute accuracy
def compute_accuracy(dataset_path):
    # Load dataset
    data = pd.read_csv(dataset_path, sep='\t')

    # Limit to the first 100 rows
    data = data.head(100)

    correct = 0
    total = len(data)

    for index, row in data.iterrows():
        premise = row['sentence_A']
        hypothesis = row['sentence_B']
        true_judgment = row['entailment_judgment'].strip().upper()  # Standardize to uppercase for comparison

        # Get NLI result from LM Studio
        nli_result = get_nli_result(premise, hypothesis)

        if nli_result:
            # Standardize the model's output to facilitate comparison
            nli_result = nli_result.strip().upper()

            # Check if the NLI result matches the true entailment judgment
            if nli_result == true_judgment:
                correct += 1

    accuracy = correct / total
    return accuracy

# Test the NLI task with the dataset
dataset_path = 'SICK_train.txt'  # Path to your dataset
accuracy = compute_accuracy(dataset_path)

print(f"Accuracy: {accuracy:.2f}")
