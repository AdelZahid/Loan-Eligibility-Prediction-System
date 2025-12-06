# Loan Eligibility Prediction System 🏦

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![Machine Learning](https://img.shields.io/badge/ML-SVM-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A web-based machine learning application that predicts loan eligibility using a trained Support Vector Machine (SVM) model. The system provides real-time predictions with an intuitive user interface.

## ✨ Features

- **🎯 Accurate Predictions**: Uses SVM model trained on historical loan data
- **🌐 Web Interface**: User-friendly form for inputting applicant details
- **📊 Real-time Results**: Instant prediction with confidence score
- **📱 Responsive Design**: Works on desktop and mobile devices
- **🔧 REST API**: Programmatic access for integration
- **🔄 Model Persistence**: Uses pickle for model storage and loading

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/loan-prediction-app.git
cd loan-prediction-app
```

### Create a virtual environment (recommended)

python -m venv venv

# Activate it

# Windows:

venv\Scripts\activate

# macOS/Linux:

source venv/bin/activate

bash
pip install -r requirements.txt
Place your trained model
Ensure your loan_status_model.pkl file is in the root directory.

Running the Application
bash
python app.py

### Project Structure

loan-prediction-app/
│
├── app.py # Main Flask application
├── loan_status_model.pkl # Trained SVM model (you must add this)
├── requirements.txt # Python dependencies
├── loan-train.csv # Dataset used for training
├── Loan_status_prediction.ipynb # Model training notebook
├── README.md # This file
│
├── templates/
│ └── index.html # Web interface
│
├── test_prediction.py # Test predictions via script
└── check_model.py # Verify model l
