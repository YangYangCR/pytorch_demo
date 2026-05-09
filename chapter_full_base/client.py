import requests

data = {
    "src": [
        [1,2,3,4,5,6,7,8,9,10]
    ],

    "tgt": [
        [11,12,13,14,15,16,17,18,19,20]
    ]
}

response = requests.post(
    "http://localhost:8000/predict",
    json=data
)

print(response.json())