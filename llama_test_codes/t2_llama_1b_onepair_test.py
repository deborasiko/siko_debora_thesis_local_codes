import json
import requests

def predict_relationship(premise, hypothesis, model="llama-3.2-1b-instruct", lm_studio_url="http://localhost:1234/v1/chat/completions"):
    messages = [
      { 
          "role": "system", 
          "content": """Analyze the relationship between these two sentences, explain the reasoning and respond with ONLY one word:
            - 'entailment' if the hypothesis logically follows from the premise
            - 'contradiction' if the hypothesis contradicts the premise
            - 'neutral' if the hypothesis is possible but not directly supported by the premise"""
        },
      { 
          "role": "user", 
          "content": f"Premise: {premise}\nHypothesis: {hypothesis}"
          }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": -1,
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
    
if __name__ == "__main__":
    premise = "The kids are playing outdoors near a man with a smile"
    hypothesis = "A group of children is playing in a yard and an old man is standing in the background"
    
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
