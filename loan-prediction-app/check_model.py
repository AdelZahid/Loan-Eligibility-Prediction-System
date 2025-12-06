# check_model.py
import pickle
import numpy as np

# Load your model
with open('loan_status_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Check model type and requirements
print(f"Model type: {type(model)}")
print(f"Model parameters: {model.get_params() if hasattr(model, 'get_params') else 'No parameters method'}")

# Try to predict with your example input
input_data = (1,0,0,6000,0,150,360,1,0,1,0)
input_data_as_numpy_array = np.array(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Check input shape
print(f"Input shape required: {input_data_reshaped.shape}")
print(f"Number of features expected: {input_data_reshaped.shape[1]}")

# Make prediction
try:
    prediction = model.predict(input_data_reshaped)
    print(f"Prediction: {prediction}")
    print(f"Prediction works correctly!")
except Exception as e:
    print(f"Error during prediction: {e}")