import aiohttp
import asyncio
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

async def get_nli_result(session, premise, hypothesis, model="gemma-3-1b-it", lm_studio_url="http://localhost:1234/v1/chat/completions"):
    headers = {"Content-Type": "application/json"}
    messages = [
        {
            "role": "system",
            "content": (
                "Classify the relationship between a premise and hypothesis.\n"
                "Respond with ONLY ONE WORD:\n"
                "- 'entailment' if the hypothesis logically follows from the premise\n"
                "- 'contradiction' if it contradicts the premise\n"
                "- 'neutral' if there's no clear relationship"
            )
        },
        {
            "role": "user",
            "content": f"Premise: {premise}\nHypothesis: {hypothesis}"
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 10,
        "stream": False
    }

    try:
        async with session.post(lm_studio_url, headers=headers, data=json.dumps(payload)) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
            else:
                print(f"Error {response.status}: {await response.text()}")
                return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

async def compute_accuracy(dataset_path, output_json_path="results/nli_predictions_neutral_gemmabase.json"):
    data = pd.read_csv(dataset_path, sep=',')

    predictions = []
    true_labels = []
    predicted_labels = []

    semaphore = asyncio.Semaphore(10)

    async def limited_task(session, premise, hypothesis):
        async with semaphore:
            return await get_nli_result(session, premise, hypothesis)

    async with aiohttp.ClientSession() as session:
        tasks = [
            limited_task(session, row['sentence_A'], row['sentence_B'])
            for _, row in data.iterrows()
        ]
        results = await asyncio.gather(*tasks)

    correct = 0
    total = len(data)

    mapping = {
        "neutral": "neutral",
        "neutrality": "neutral",
        "entailment": "entailment",
        "contradiction": "contradiction",
    }

    for idx, row in enumerate(data.itertuples(index=False)):
        true_label = row.entailment_judgment.strip().lower()
        predicted_label_raw = (results[idx] or "").strip().lower().strip(".-*!\n ")

        # Default to unknown
        cleaned_pred = "unknown"
        for key in mapping:
            if key in predicted_label_raw:
                cleaned_pred = mapping[key]
                break

        true_labels.append(true_label)
        predicted_labels.append(cleaned_pred)

        print(f"Example {idx+1}: True: {true_label} | Predicted: {cleaned_pred}")

        if cleaned_pred == true_label:
            correct += 1

        predictions.append({
            "premise": row.sentence_A,
            "hypothesis": row.sentence_B,
            "true_label": true_label,
            "predicted_label": cleaned_pred
        })

    # Save predictions to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    accuracy = correct / total if total > 0 else 0

    # Create confusion matrix and plot
    label_order = ["contradiction", "neutral", "entailment"]
    confusion_matrix = pd.crosstab(
        pd.Categorical(true_labels, categories=label_order, ordered=True),
        pd.Categorical(predicted_labels, categories=label_order, ordered=True)
    )

    if confusion_matrix.empty or confusion_matrix.sum().sum() == 0:
        print("Confusion matrix is empty or contains only zeros. Skipping heatmap.")
    else:
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title("Finetuned NLI Model - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

    return accuracy

# Run evaluation
if __name__ == "__main__":
    dataset_path = "data_files/neutral_sick_data.txt"
    accuracy = asyncio.run(compute_accuracy(dataset_path))
    print(f"\nFinal Accuracy: {accuracy:.2f}")
