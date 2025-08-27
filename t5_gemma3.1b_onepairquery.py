import requests
import json

def predict_relationship(premise, hypothesis, model="gemma-3-1b-it", lm_studio_url="http://localhost:1234/v1/chat/completions"):
    """
    Makes an API call to LM Studio's chat completions endpoint to predict the relationship between sentences.
    
    Args:
        premise (str): The premise sentence
        hypothesis (str): The hypothesis sentence
        model (str): The model name to use (default: "gemma-3-1b-it")
        lm_studio_url (str): The URL of your LM Studio API endpoint
        
    Returns:
        dict: The model's response containing the predicted relationship
    """
    
    # Prepare the system message and user prompt
    messages = [
        {
            "role": "system",
            "content": """Analyze the relationship between these two sentences, explain the reasoning and after explaining respond with ONLY one word:
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
        "max_tokens": 10,    # Limit response length
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

# Example usage
if __name__ == "__main__":
    premise = "The young boys are playing outdoors and the man is smiling nearby"
    hypothesis = "The kids are playing outdoors near a man with a smile"
    
    result = predict_relationship(premise, hypothesis)
    
    if result:
        print("Full API response:")
        print(json.dumps(result, indent=2))
        
        # Extract the prediction from the chat response
        prediction = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip().lower()
        print(f"\nPremise: {premise}")
        print(f"\nHypothesis: {hypothesis}")
        print(f"\nPredicted relationship: {prediction}")
    else:
        print("Failed to get prediction")