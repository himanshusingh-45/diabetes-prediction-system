import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

df = pd.read_csv(url, names=columns)

# 2. Preprocessing: Clean missing zero values using modern Pandas syntax (No Warnings)
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    # Replaced inplace=True with direct reassignment to prevent Pandas ChainedAssignmentError
    df[col] = df[col].fillna(df[col].median())

# 3. Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Optimized Model Training
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=4,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 7. Model Evaluation
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"✨ Clean Run - Model Accuracy: {acc * 100:.2f}%")

# 8. Save Updated Model Files
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("💾 Updated diabetes_model.pkl and scaler.pkl successfully!")