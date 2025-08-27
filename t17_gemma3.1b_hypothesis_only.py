import aiohttp
import asyncio
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

async def get_nli_result(session, premise, hypothesis, model="gemma-3-1b-it", lm_studio_url="http://localhost:1234/v1/chat/completions", retries=3):
    """
    Sends premise and hypothesis to the LM Studio API for NLI prediction asynchronously.
    Retries the request if it fails.
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

    for attempt in range(retries):
        try:
            async with session.post(lm_studio_url, headers=headers, data=json.dumps(payload)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
                else:
                    text = await response.text()
                    print(f"[Attempt {attempt+1}] Error {response.status}: {text}")
        except Exception as e:
            print(f"[Attempt {attempt+1}] Exception: {e}")
        await asyncio.sleep(1)  # Small delay before retry
    return None  # If all retries fail

async def compute_accuracy(dataset_path):
    """
    Reads the dataset and evaluates the model's accuracy using asynchronous requests.
    Also prints the predicted and true labels and generates a heatmap.
    """
    data = pd.read_csv(dataset_path, sep=',')
    true_labels = []
    predicted_labels = []

    semaphore = asyncio.Semaphore(10)  # Concurrency limit

    async def limited_get_nli_result(session, premise, hypothesis):
        async with semaphore:
            return await get_nli_result(session, premise, hypothesis)

    async with aiohttp.ClientSession() as session:
        tasks = [
            limited_get_nli_result(session, row['sentence_A'], row['sentence_B']) 
            for _, row in data.iterrows()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    correct = 0
    total = len(data)

    for idx, (row, result) in enumerate(zip(data.itertuples(index=False), results)):
        true_judgment = row.entailment_judgment.strip().lower()

        if isinstance(result, Exception) or result is None:
            predicted_judgment = ""
            print(f"Example {idx + 1}: API failed. True Label: {true_judgment}, Predicted: None")
        else:
            predicted_judgment = result.strip().lower()
            print(f"Example {idx + 1}: True Label: {true_judgment}, Predicted: {predicted_judgment}")

        true_labels.append(true_judgment)
        predicted_labels.append(predicted_judgment)

        if predicted_judgment == true_judgment and predicted_judgment != "":
            correct += 1

    accuracy = correct / total if total else 0.0

    # Create confusion matrix
    label_order = ["contradiction", "neutral", "entailment"]
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

# Run the evaluation
dataset_path = 'only_hypothesis.txt'
accuracy = asyncio.run(compute_accuracy(dataset_path))
print(f"Model Accuracy: {accuracy:.2f}")
