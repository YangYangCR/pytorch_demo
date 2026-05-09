import requests


response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "hello world"
    }
)

print(response.json())