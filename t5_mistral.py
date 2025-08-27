import requests

url = "http://192.168.100.44:1234/v1/chat/completions"  
headers = {"Content-Type": "application/json"}

data_payload = {
    "model": "mistral-7b-instruct-v0.3",
    "messages": [
        {"role": "user", "content": "Hello! How are you?"}
    ],
    "temperature": 0.5
}

response = requests.post(url, headers=headers, json=data_payload)

print("Status Code:", response.status_code)
print("Full Response:", response.text)
