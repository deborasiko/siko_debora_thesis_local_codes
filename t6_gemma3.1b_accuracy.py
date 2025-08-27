import requests
import pandas as pd
import json

def get_nli_result(premise, hypothesis, model="gemma-3-1b-it", lm_studio_url="http://localhost:1234/v1/chat/completions"):
    """
    Sends premise and hypothesis to the LM Studio API for NLI prediction.
    """
    headers = {"Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": """Analyze the relationship between these two sentences and respond with ONLY one word:
        - 'entailment' if the hypothesis logically follows from the premise
        - 'contradiction' if the hypothesis contradicts the premise
        - 'neutral' if there's no clear relationship"""},
        {"role": "user", "content": f"Premise: {premise}\nHypothesis: {hypothesis}"}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 10,
        "stream": False
    }
    
    response = requests.post(lm_studio_url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
    else:
        print(f"Error: {response.status_code}")
        return None

def compute_accuracy(dataset_path):
    """
    Reads the dataset and evaluates the model's accuracy.
    """
    data = pd.read_csv(dataset_path, sep='\t')
    # data = data.head(100)  # Limit to 100 samples

    correct = 0
    total = len(data)
    
    for index, row in data.iterrows():
        premise = row['sentence_A']
        hypothesis = row['sentence_B']
        true_judgment = row['entailment_judgment'].strip().lower()
        
        predicted_judgment = get_nli_result(premise, hypothesis)
        
        if predicted_judgment and predicted_judgment == true_judgment:
            correct += 1
    
    accuracy = correct / total
    return accuracy

# Run evaluation
dataset_path = 'SICK_train.txt'  # Adjust the path if necessary
accuracy = compute_accuracy(dataset_path)
print(f"Model Accuracy: {accuracy:.2f}")
