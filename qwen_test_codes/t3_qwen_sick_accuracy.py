import requests
import json
import json
import requests
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def predict_relationship(premise, hypothesis, model="qwen3-0.6b", lm_studio_url="http://localhost:1234/v1/chat/completions"):
    
    # Prepare the system message and user prompt
    messages = [
        {
            "role": "system",
            "content": """Respond with ONLY one word:
            - 'entailment' if the hypothesis logically follows from the premise
            - 'contradiction' if the hypothesis contradicts the premise
            - 'neutral' if there's no clear relationship"""
        },
        {
            "role": "user",
            "content": f"Premise: {premise}\nHypothesis: {hypothesis}"
        }
    ]
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,  # Lower temperature for more deterministic output
        "max_tokens": -1,    # Limit response length
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Make the API request
        response = requests.post(lm_studio_url, 
                               data=json.dumps(payload), 
                               headers=headers)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    
def normalize_prediction(pred):
    pred = pred.lower().strip().strip(".-*!\n ")
    mapping = {
        "neutral": "neutral",
        "neutrality": "neutral",
        "entailment": "entailment",
        "contradiction": "contradiction",
    }
    for key in mapping:
        if key in pred:
            return mapping[key]
    return "unknown"

def compute_accuracy(dataset_path, output_json_path = "nli_predictions_qwen.json"):
    data = pd.read_csv(dataset_path, sep='\t')
    # data = data.head(10)
    #data = data.head(200)  # Process first 200 samples

    true_labels = []
    predicted_labels = []
    total = len(data)
    correct = 0
    results = []  # List to hold per-example results

    for _, row in data.iterrows():
        premise = row['sentence_A']
        hypothesis = row['sentence_B']
        true_judgment = row['entailment_judgment'].strip().lower()
        
        predicted_judgment = predict_relationship(premise, hypothesis)
        response = predict_relationship(premise, hypothesis)
        raw_prediction = response["choices"][0]["message"]["content"]
        cleaned_prediction = raw_prediction.split()[-1]
        cleaned_prediction = normalize_prediction(raw_prediction)

        true_labels.append(true_judgment)
        predicted_labels.append(cleaned_prediction)

        if cleaned_prediction == true_judgment:
            correct += 1

        # Store the result for JSON output
        results.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "true_label": true_judgment,
            "predicted_label": cleaned_prediction
        })
         # Save to JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    accuracy = correct / total
    label_order = ["contradiction", "neutral", "entailment"]
    # Create a confusion matrix
    confusion_matrix = pd.crosstab(
            pd.Categorical(true_labels, categories=label_order, ordered=True),
            pd.Categorical(predicted_labels, categories=label_order, ordered=True)
            )

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("NLI Prediction Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    return accuracy

# Run evaluation
dataset_path = '../SICK_train.txt'  # Adjust the path if necessary
accuracy = compute_accuracy(dataset_path)
print(f"Model Accuracy: {accuracy:.2f}")