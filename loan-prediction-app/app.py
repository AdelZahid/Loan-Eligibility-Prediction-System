# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
try:
    with open('loan_status_model.pkl', 'rb') as file:
        classifier = pickle.load(file)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

# Define feature mapping based on your input (1,0,0,6000,0,150,360,1,0,1,0)
# You'll need to adjust this based on how your data was preprocessed
FEATURE_MAPPING = [
    {"name": "Gender", "options": ["Male", "Female"], "mapping": {"Male": 1, "Female": 0}},
    {"name": "Married", "options": ["No", "Yes"], "mapping": {"No": 0, "Yes": 1}},
    {"name": "Dependents", "options": ["0", "1", "2", "3+"], "mapping": {"0": 0, "1": 1, "2": 2, "3+": 3}},
    {"name": "Education", "options": ["Graduate", "Not Graduate"], "mapping": {"Graduate": 1, "Not Graduate": 0}},
    {"name": "Self_Employed", "options": ["No", "Yes"], "mapping": {"No": 0, "Yes": 1}},
    {"name": "ApplicantIncome", "type": "number", "min": 0, "max": 100000},
    {"name": "CoapplicantIncome", "type": "number", "min": 0, "max": 50000},
    {"name": "LoanAmount", "type": "number", "min": 0, "max": 1000},
    {"name": "Loan_Amount_Term", "type": "number", "min": 12, "max": 480},
    {"name": "Credit_History", "options": ["Good (1)", "Bad (0)"], "mapping": {"Good (1)": 1, "Bad (0)": 0}},
    {"name": "Property_Area", "options": ["Urban", "Semiurban", "Rural"], "mapping": {"Urban": 0, "Semiurban": 1, "Rural": 2}}
]

@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_MAPPING)

@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None:
        return jsonify({'error': 'Model not loaded. Please check loan_status_model.pkl file.'})
    
    try:
        # Get form data
        data = request.form
        
        # Prepare input array based on your example: (1,0,0,6000,0,150,360,1,0,1,0)
        input_features = []
        
        # Gender (Male=1, Female=0)
        gender = data.get('gender', 'Male')
        input_features.append(1 if gender == 'Male' else 0)
        
        # Married (No=0, Yes=1)
        married = data.get('married', 'No')
        input_features.append(1 if married == 'Yes' else 0)
        
        # Dependents (0, 1, 2, 3+)
        dependents = data.get('dependents', '0')
        if dependents == '0':
            input_features.append(0)
        elif dependents == '1':
            input_features.append(1)
        elif dependents == '2':
            input_features.append(2)
        else:  # 3+
            input_features.append(3)
        
        # Education (Graduate=1, Not Graduate=0)
        education = data.get('education', 'Graduate')
        input_features.append(1 if education == 'Graduate' else 0)
        
        # Self Employed (No=0, Yes=1)
        self_employed = data.get('self_employed', 'No')
        input_features.append(1 if self_employed == 'Yes' else 0)
        
        # ApplicantIncome (keep as is)
        applicant_income = float(data.get('applicant_income', 0))
        input_features.append(applicant_income)
        
        # CoapplicantIncome (keep as is)
        coapplicant_income = float(data.get('coapplicant_income', 0))
        input_features.append(coapplicant_income)
        
        # LoanAmount (keep as is)
        loan_amount = float(data.get('loan_amount', 0))
        input_features.append(loan_amount)
        
        # Loan_Amount_Term (keep as is)
        loan_term = float(data.get('loan_term', 360))
        input_features.append(loan_term)
        
        # Credit_History (1=Good, 0=Bad)
        credit_history = data.get('credit_history', 'Good (1)')
        input_features.append(1 if credit_history == 'Good (1)' else 0)
        
        # Property_Area (Urban=0, Semiurban=1, Rural=2)
        property_area = data.get('property_area', 'Urban')
        if property_area == 'Urban':
            input_features.append(0)
        elif property_area == 'Semiurban':
            input_features.append(1)
        else:  # Rural
            input_features.append(2)
        
        # Convert to numpy array and reshape
        input_array = np.array(input_features).reshape(1, -1)
        
        # Debug: Print input array
        print(f"Input array: {input_array}")
        
        # Make prediction
        prediction = classifier.predict(input_array)
        
        # Get prediction probability if available
        if hasattr(classifier, 'predict_proba'):
            probability = classifier.predict_proba(input_array)[0]
            approval_prob = probability[1] if len(probability) > 1 else probability[0]
        else:
            approval_prob = 0.8 if prediction[0] == 1 else 0.2
        
        # Prepare response
        result = {
            'prediction': int(prediction[0]),
            'loan_status': 'Approved' if prediction[0] == 1 else 'Not Approved',
            'approval_probability': float(approval_prob),
            'confidence': f"{approval_prob * 100:.1f}%",
            'input_features': input_features,
            'details': {
                'recommendation': ' Congratulations! You are eligible for the loan.' if prediction[0] == 1 
                            else ' Sorry, you are not eligible for the loan.',
                'suggestion': 'You can proceed with the loan application.' if prediction[0] == 1
                            else 'Consider improving your credit score or reducing the loan amount.'
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': 'Check server logs for details'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for raw data input"""
    if classifier is None:
        return jsonify({'error': 'Model not loaded'})
    
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'No features provided'})
        
        # Convert to numpy array
        input_array = np.array(data['features']).reshape(1, -1)
        
        # Make prediction
        prediction = classifier.predict(input_array)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'loan_status': 'Approved' if prediction[0] == 1 else 'Not Approved'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/features')
def get_features():
    """Return feature information"""
    return jsonify({
        'feature_count': len(FEATURE_MAPPING),
        'features': FEATURE_MAPPING,
        'example_input': [1, 0, 0, 6000, 0, 150, 360, 1, 0, 1, 0],
        'prediction_mapping': {
            '1': 'Eligible for loan',
            '0': 'Not eligible for loan'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)