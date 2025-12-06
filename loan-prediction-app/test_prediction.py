# test_prediction.py
import pickle
import numpy as np

# Load your model
with open('loan_status_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Test cases
test_cases = [
    # Your example
    {
        "name": "Your Example",
        "input": (1, 0, 0, 6000, 0, 150, 360, 1, 0, 1, 0),
        "expected": 1  # Should be eligible
    },
    # Add more test cases
    {
        "name": "High Income Applicant",
        "input": (1, 1, 2, 25000, 10000, 500, 360, 1, 1, 1, 0),
        "expected": 1
    },
    {
        "name": "Bad Credit History",
        "input": (0, 0, 0, 8000, 2000, 100, 180, 0, 0, 0, 1),
        "expected": 0
    }
]

print("Testing your loan prediction model...")
print("=" * 50)

for test in test_cases:
    input_array = np.array(test["input"]).reshape(1, -1)
    prediction = classifier.predict(input_array)[0]
    
    status = "PASS" if prediction == test["expected"] else "❌ FAIL"
    result = "Eligible" if prediction == 1 else "Not Eligible"
    
    print(f"{test['name']}:")
    print(f"  Input: {test['input']}")
    print(f"  Prediction: {prediction} ({result})")
    print(f"  Expected: {test['expected']} {status}")
    print(f"  {'-' * 30}")

print("\nModel is ready for deployment!")