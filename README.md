
# Healthcare Digital Twin - Explainable AI

## Project Title
A Trust-Aware and Explainable Healthcare Digital Twin
for Diabetes Risk Prediction Using SHAP

## Introduction
This project develops a simplified healthcare digital
twin using machine learning to predict the diabetes
dataset outcome and Explainable AI to interpret
individual predictions.

## Problem Statement
Machine learning healthcare prediction systems may
produce outputs without clear explanations. This
project uses SHAP to improve model transparency
and demonstrate explainability.

## Objectives
- Develop a virtual patient profile
- Predict diabetes dataset outcomes
- Explain predictions using SHAP
- Display results through Streamlit
- Demonstrate transparency in AI

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Streamlit
- Matplotlib

## Project Structure
- data/diabetes.csv
- train_model.py
- app.py
- requirements.txt
- README.md

## Installation

1. Clone this repository.
2. Create a Python virtual environment.
3. Install dependencies.

```bash
pip install -r requirements.txt
```

## Run the Project

Train the model:

```bash
python train_model.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

## Explainability
SHAP is used to display the contribution of input
features to the model prediction.

## Disclaimer
This project is for educational purposes only.
It is not a clinical diagnostic system.

## Dataset
Pima Indians Diabetes Dataset.
Original source: UCI Machine Learning Repository.

## Author
ADITYA KUMAR
2nd Year Innovative Project
