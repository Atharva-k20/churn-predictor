import requests

data = {
    'tenure': 2,
    'MonthlyCharges': 80,
    'Contract_Month-to-month': 1
}

r = requests.post('http://localhost:5000/predict', json=data)
print(r.json())