import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD TRAINING DATA
# ==========================================

train_data = pd.read_csv("Training.csv")

# Clean disease names
train_data["prognosis"] = (
    train_data["prognosis"]
    .astype(str)
    .str.strip()
)

# Input features and target
X_train = train_data.drop("prognosis", axis=1)
y_train = train_data["prognosis"]


# ==========================================
# 2. LOAD TESTING DATA
# ==========================================

test_data = pd.read_csv("Testing.csv")

# Clean disease names
test_data["prognosis"] = (
    test_data["prognosis"]
    .astype(str)
    .str.strip()
)

X_test = test_data.drop("prognosis", axis=1)
y_test = test_data["prognosis"]


# ==========================================
# 3. MAKE SURE FEATURES ARE IN SAME ORDER
# ==========================================

X_test = X_test[X_train.columns]


# ==========================================
# 4. TRAIN RANDOM FOREST
# ==========================================

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# ==========================================
# 5. TEST MODEL
# ==========================================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(
    "Random Forest Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ==========================================
# 6. CLASSIFICATION REPORT
# ==========================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_test,
        prediction,
        zero_division=0
    )
)


# ==========================================
# 7. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/disease_model.pkl"
)

joblib.dump(
    list(X_train.columns),
    "models/features.pkl"
)

print("\n================================")
print("MODEL SAVED")
print("================================")

print("Model: models/disease_model.pkl")
print("Features: models/features.pkl")
print("Training completed successfully!")