import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)

import joblib

warnings.filterwarnings("ignore")

# ==================================================
# CREATE FOLDERS
# ==================================================
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ==================================================
# LOAD DATASET
# ==================================================
try:
    df = pd.read_csv("data/dataset.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("ERROR: data/dataset.csv not found.")
    exit()

# ==================================================
# DISPLAY BASIC INFO
# ==================================================
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ==================================================
# REMOVE MISSING VALUES
# ==================================================
df = df.dropna()

# ==================================================
# ENCODE GENDER
# ==================================================
if "Gender" in df.columns:
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

# ==================================================
# FEATURES AND TARGET
# ==================================================
X = df.drop("Pass", axis=1)
y = df["Pass"]

# ==================================================
# TRAIN TEST SPLIT
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==================================================
# RANDOM FOREST MODEL
# ==================================================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==================================================
# PREDICTIONS
# ==================================================
y_pred = model.predict(X_test)

# ==================================================
# ACCURACY
# ==================================================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:")
print(round(accuracy * 100, 2), "%")

# ==================================================
# CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "images/confusion_matrix.png"
)

plt.close()

# ==================================================
# CLASSIFICATION REPORT
# ==================================================
report_text = classification_report(
    y_test,
    y_pred
)

print("\nClassification Report:")
print(report_text)

# ==================================================
# ROC CURVE
# ==================================================
try:

    y_prob = model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(
        y_test,
        y_prob
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.2f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "images/roc_curve.png"
    )

    plt.close()

except Exception as e:
    print("ROC Curve skipped:", e)
    auc_score = 0

# ==================================================
# ACTUAL VS PREDICTED
# ==================================================
plt.figure(figsize=(8, 5))

plt.plot(
    range(len(y_test)),
    list(y_test),
    marker="o",
    label="Actual"
)

plt.plot(
    range(len(y_pred)),
    list(y_pred),
    marker="x",
    label="Predicted"
)

plt.title("Actual vs Predicted")

plt.xlabel("Sample")

plt.ylabel("Pass / Fail")

plt.legend()

plt.tight_layout()

plt.savefig(
    "images/actual_vs_predicted.png"
)

plt.close()

# ==================================================
# SAVE MODEL
# ==================================================
joblib.dump(
    model,
    "models/predictive_model.pkl"
)

# ==================================================
# SAVE REPORT
# ==================================================
with open(
    "reports/Model_Report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PREDICTIVE MODELING REPORT\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Dataset Shape: {df.shape}\n\n"
    )

    file.write(
        f"Accuracy Score: {accuracy:.4f}\n\n"
    )

    file.write(
        f"AUC Score: {auc_score:.4f}\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        report_text
    )

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")
print("models/predictive_model.pkl")
print("images/confusion_matrix.png")
print("images/roc_curve.png")
print("images/actual_vs_predicted.png")
print("reports/Model_Report.txt")