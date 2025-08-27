import aiohttp
import asyncio
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

async def get_nli_result(session, premise, hypothesis, model="gemma-3-1b-it", lm_studio_url="http://localhost:1234/v1/chat/completions"):
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
    data = pd.read_csv(dataset_path, sep=',')

    true_labels = []
    predicted_labels = []

    semaphore = asyncio.Semaphore(10)

    async def limited_get_nli_result(session, premise, hypothesis):
        async with semaphore:
            return await get_nli_result(session, premise, hypothesis)

    async with aiohttp.ClientSession() as session:
        tasks = [
            limited_get_nli_result(session, row['sentence_A'], row['sentence_B']) 
            for _, row in data.iterrows()
        ]
        results = await asyncio.gather(*tasks)

    correct = 0
    total = len(data)
    
    label_mapping = {
        "neutral": "neutral",
        "neutrality": "neutral",
        "entailment": "entailment",
        "contradiction": "contradiction"
    }

    for idx, row in enumerate(data.itertuples(index=False)):
        true_judgment = row.entailment_judgment.strip().lower()
        predicted_judgment = (results[idx] or "").lower().strip().strip(".-*!\n ")
        cleaned_pred = label_mapping.get(predicted_judgment, "unknown")

        true_labels.append(true_judgment)
        predicted_labels.append(cleaned_pred)
        
        print(f"Example {idx+1}: True Label: {true_judgment}, Predicted: {cleaned_pred}")

        if cleaned_pred == true_judgment:
            correct += 1

    print("Prediction counts:", Counter(predicted_labels))

    accuracy = correct / total if total > 0 else 0

    label_order = ["contradiction", "neutral", "entailment"]
    confusion_matrix = pd.crosstab(
        pd.Categorical(true_labels, categories=label_order, ordered=True),
        pd.Categorical(predicted_labels, categories=label_order, ordered=True)
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("NLI Prediction Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    return accuracy


# Run evaluation
dataset_path = 'data_files/entailment_sick_data.txt'
accuracy = asyncio.run(compute_accuracy(dataset_path))
print(f"Model Accuracy: {accuracy:.2f}")
