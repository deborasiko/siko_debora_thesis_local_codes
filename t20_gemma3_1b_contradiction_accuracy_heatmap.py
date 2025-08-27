import aiohttp
import asyncio
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

async def get_nli_result(session, premise, hypothesis, model="gemma-3-1b-it", lm_studio_url="http://localhost:1234/v1/chat/completions"):
    """
    Sends premise and hypothesis to the LM Studio API for NLI prediction asynchronously.
    """
    headers = {"Content-Type": "application/json"}
    messages = [
        {"role": "system", 
         "content": (
                "Classify the relationship between a premise and hypothesis. "
                "Respond with ONLY ONE WORD:\n"
                "- 'entailment' if the hypothesis logically follows from the premise\n"
                "- 'contradiction' if it contradicts the premise\n"
                "- 'neutral' if there's no clear relationship"
            )}, 
        {"role": "user", "content": f"Premise: {premise}\nHypothesis: {hypothesis}"}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 10,
        "stream": False
    }

    async with session.post(lm_studio_url, headers=headers, data=json.dumps(payload)) as response:
        if response.status == 200:
            result = await response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
        else:
            print(f"Error: {response.status}")
            return None



async def compute_accuracy(dataset_path):
    """
    Reads the dataset and evaluates the model's accuracy using asynchronous requests.
    Also prints the predicted and true labels and generates a heatmap.
    """
    data = pd.read_csv(dataset_path, sep=',')
    # data = data.head(200)  # Process first 200 samples

    true_labels = []
    predicted_labels = []

    semaphore = asyncio.Semaphore(10)  # Adjust concurrency limit as needed

    async def limited_get_nli_result(session, premise, hypothesis):
        async with semaphore:  # Limit concurrent requests
            return await get_nli_result(session, premise, hypothesis)

    async with aiohttp.ClientSession() as session:
        tasks = [
            limited_get_nli_result(session, row['sentence_A'], row['sentence_B']) 
            for _, row in data.iterrows()
        ]
        results = await asyncio.gather(*tasks)

    correct = 0
    total = len(data)
    
    for idx, row in enumerate(data.itertuples(index=False)):
        true_judgment = row.entailment_judgment.strip().lower()
        # predicted_judgment = results[idx]
        predicted_judgment = (results[idx] or "").strip().lower()
        cleaned_pred = predicted_judgment.lower().strip().strip(".-*!\n ")
        mapping = {
                    "neutral": "neutral",
                    "neutrality": "neutral",
                    "entailment": "entailment",
                    "contradiction": "contradiction",
                }
        for key in mapping:
            if key in cleaned_pred:
                cleaned_pred = mapping[key]
            cleaned_pred = "unknown"
        true_labels.append(true_judgment)
        predicted_labels.append(cleaned_pred)
        
        print(f"Example {idx+1}: True Label: {true_judgment}, Predicted: {predicted_judgment}")
        
        if predicted_judgment and predicted_judgment == true_judgment:
            correct += 1
    
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
dataset_path = 'data_files/contradiction_sick_data.txt'
accuracy = asyncio.run(compute_accuracy(dataset_path))
print(f"Model Accuracy: {accuracy:.2f}")
