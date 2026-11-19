# BIOLOGY FOR ENGINEERS
# Diabetes Prediction Using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, RocCurveDisplay
)

# 1. LOAD DATASET
url = (
    "https://raw.githubusercontent.com/"
    "jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

columns = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(url, header=None, names=columns)

print("Dataset shape:", df.shape)
print(df.head())
print("\nClass distribution:")
print(df["Outcome"].value_counts())

# 2. PREPROCESS DATA
X = df.drop("Outcome", axis=1).copy()
y = df["Outcome"]

# Zero values are invalid for these measurements
invalid_zero_cols = [
    "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI"
]

X[invalid_zero_cols] = X[invalid_zero_cols].replace(
    0, np.nan
)

# 3. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. CREATE MODELS
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])
}

# 5. TRAIN AND EVALUATE
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    print("\nMODEL:", name)
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.title("Confusion Matrix - " + name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        name.replace(" ", "_") + "_confusion.png",
        dpi=300
    )
    plt.show()

# 6. MODEL COMPARISON
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON:")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "diabetes_model_results.csv", index=False
)

# 7. PERFORMANCE COMPARISON GRAPH
metrics = [
    "Accuracy", "Precision", "Recall",
    "F1 Score", "ROC AUC"
]

results_df.set_index("Model")[metrics].T.plot(
    kind="bar", figsize=(10, 6)
)

plt.title("Comparison of ML Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()

# 8. FEATURE IMPORTANCE
rf = models["Random Forest"].named_steps["model"]

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values()

importance.plot(kind="barh", figsize=(8, 5))
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

# 9. ROC CURVES
fig, ax = plt.subplots(figsize=(7, 6))

for name, model in models.items():
    RocCurveDisplay.from_estimator(
        model, X_test, y_test,
        name=name, ax=ax
    )

plt.title("ROC Curve Comparison")
plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=300)
plt.show()

print("\nExperiment completed successfully!")