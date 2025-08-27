import requests
import json

def predict_relationship_finetuned(premise, hypothesis, model="gemma-3-1b-it-finetuned", lm_studio_url="http://localhost:1234/v1/chat/completions"):
    """
    Predict the relationship between a premise and hypothesis using a fine-tuned LLM from LM Studio.

    Args:
        premise (str): The premise sentence.
        hypothesis (str): The hypothesis sentence.
        model (str): Name of the finetuned model loaded in LM Studio.
        lm_studio_url (str): URL to the LM Studio API.

    Returns:
        str: One of 'entailment', 'contradiction', or 'neutral'.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Classify the relationship between a premise and hypothesis. "
                "Reply with ONLY ONE WORD:\n"
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

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(lm_studio_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        result = response.json()

        # Extract and normalize the prediction
        prediction = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip().lower()
        return prediction

    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    premise = "The man is playing a guitar at a concert."
    hypothesis = "A person is performing music in front of a crowd."

    prediction = predict_relationship_finetuned(premise, hypothesis)
    
    print(f"Premise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Predicted relationship: {prediction}")
