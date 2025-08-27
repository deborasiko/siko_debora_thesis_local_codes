import requests
import pandas as pd

url = "http://192.168.100.44:1234/v1/chat/completions"

# Headers for API request
headers = {"Content-Type": "application/json"}

data = pd.read_csv('SICK_train.txt', sep='\t')

# Select a single pair
premise = data.loc[12, 'sentence_A']
hypothesis = data.loc[12, 'sentence_B']
true_label = data.loc[12, 'entailment_judgment']


def get_nli_prediction(premise, hypothesis):

    data_payload = {
        "model": "mistral-7b-instruct-v0.3",  
        "messages": [
            {"role": "user", "content": f"Premise: {premise}\nHypothesis: {hypothesis}\nWhat is the relationship? (Entailment, Neutral, or Contradiction?)"}
        ],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=data_payload)
    
    if response.status_code == 200:
        response_json = response.json()
        if "choices" in response_json:
            model_output = response_json["choices"][0]["message"]["content"].lower()
            print(response.json())
            if "entailment" in model_output:
                return "ENTAILMENT"
            elif "neutral" in model_output:
                return "NEUTRAL"
            elif "contradiction" in model_output:
                return "CONTRADICTION"
            else:
                return "UNKNOWN"
        else:
            return "ERROR: 'choices' key missing"
    else:
        return f"ERROR: Status {response.status_code}"

predicted_label = get_nli_prediction(premise, hypothesis)

print(f"\nPremise: {premise}")
print(f"Hypothesis: {hypothesis}")
print(f"True Label: {true_label}")
print(f"Predicted Label: {predicted_label}\n")