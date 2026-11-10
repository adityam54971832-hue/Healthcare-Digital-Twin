

import pandas as pd

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv("data/diabetes.csv", header=None, names=columns)

# Save the corrected dataset
df.to_csv("data/diabetes.csv", index=False)

print("Dataset corrected successfully!")
print("Dataset shape:", df.shape)
print(df.head())
print(df.columns.tolist())
