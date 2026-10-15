"""
Disease prediction model (CodeAlpha internship, Task 4)

This script uses the Breast Cancer Wisconsin (Diagnostic) dataset.
Each row is one patient, and the columns are measurements taken from
images of a cell sample. The goal is to predict if the tumour is
cancerous (malignant) or not (benign).

"Positive" in every metric below means the tumour is MALIGNANT,
because missing a sick patient is the worst mistake here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

RESULTS_FOLDER = Path("results")
DATA_FILE = Path("data") / "data.csv"
RANDOM_SEED = 42
NUMBER_OF_CROSS_VALIDATION_FOLDS = 5


def load_patient_data():
    patient_table = pd.read_csv(DATA_FILE)

    # The file has an empty column at the end (from a comma at the end
    # of the header) and an id column. Neither helps the model, and the
    # id would only be noise.
    patient_table = patient_table.dropna(axis="columns", how="all")
    patient_table = patient_table.drop(columns=["id"])

    # In the file M means malignant and B means benign.
    # I turn it into 1 = has the disease, 0 = does not.
    tumour_is_malignant = (patient_table["diagnosis"] == "M").astype(int)
    patient_measurements = patient_table.drop(columns=["diagnosis"])
    return patient_measurements, tumour_is_malignant


def build_models():
    # Logistic Regression and SVM need scaled numbers, so they get a
    # scaler inside their own pipeline. The tree models do not need it.
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "SVM": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        ),
    }


def check_model_with_cross_validation(model, train_measurements, train_answers):
    # Training data is split into 5 parts. The model trains on 4 parts and
    # is checked on the 5th, and this repeats 5 times. It gives a more
    # stable idea of the recall than one single split.
    fold_splitter = StratifiedKFold(
        n_splits=NUMBER_OF_CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    recall_per_fold = cross_val_score(
        model, train_measurements, train_answers, cv=fold_splitter, scoring="recall"
    )
    return recall_per_fold.mean(), recall_per_fold.std()


def evaluate_on_test_data(model_name, trained_model, test_measurements, test_answers):
    predicted_labels = trained_model.predict(test_measurements)
    predicted_malignant_chance = trained_model.predict_proba(test_measurements)[:, 1]

    print(f"\n--- {model_name} ---")
    print("Confusion matrix (rows = real, columns = predicted):")
    print(confusion_matrix(test_answers, predicted_labels))

    # Sick patients that the model called healthy.
    missed_sick_patients = ((test_answers == 1) & (predicted_labels == 0)).sum()
    print(f"Sick patients the model missed: {missed_sick_patients}")

    scores = {
        "Model": model_name,
        "Accuracy": accuracy_score(test_answers, predicted_labels),
        "Precision": precision_score(test_answers, predicted_labels),
        "Recall": recall_score(test_answers, predicted_labels),
        "F1-Score": f1_score(test_answers, predicted_labels),
        "ROC-AUC": roc_auc_score(test_answers, predicted_malignant_chance),
        "Missed sick patients": int(missed_sick_patients),
    }
    return scores, predicted_malignant_chance


def save_roc_curve_picture(test_answers, chances_by_model):
    plt.figure(figsize=(6, 5))
    for model_name, chances in chances_by_model.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(test_answers, chances)
        plt.plot(false_positive_rate, true_positive_rate, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guessing")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on the test data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_FOLDER / "roc_curves.png", dpi=150)
    plt.close()


def save_forest_importance_picture(random_forest_model, measurement_names):
    top_ten = (
        pd.Series(random_forest_model.feature_importances_, index=measurement_names)
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop 10 measurements for the Random Forest:")
    print(top_ten)

    plt.figure(figsize=(8, 5))
    top_ten.sort_values().plot(kind="barh")
    plt.title("Top 10 most important measurements (Random Forest)")
    plt.tight_layout()
    plt.savefig(RESULTS_FOLDER / "feature_importance.png", dpi=150)
    plt.close()


def main():
    RESULTS_FOLDER.mkdir(exist_ok=True)

    patient_measurements, tumour_is_malignant = load_patient_data()
    print(f"Loaded {len(patient_measurements)} patients with "
          f"{patient_measurements.shape[1]} measurements each.")
    print("Share of malignant tumours:", round(tumour_is_malignant.mean(), 3))

    train_measurements, test_measurements, train_answers, test_answers = train_test_split(
        patient_measurements,
        tumour_is_malignant,
        test_size=0.2,
        stratify=tumour_is_malignant,
        random_state=RANDOM_SEED,
    )

    all_scores = []
    chances_by_model = {}
    trained_models = {}

    for model_name, model in build_models().items():
        average_recall, recall_spread = check_model_with_cross_validation(
            model, train_measurements, train_answers
        )
        print(f"\n{model_name}: cross-validated recall = "
              f"{average_recall:.3f} (+/- {recall_spread:.3f})")

        model.fit(train_measurements, train_answers)
        trained_models[model_name] = model

        scores, chances = evaluate_on_test_data(
            model_name, model, test_measurements, test_answers
        )
        scores["Cross-validated recall"] = average_recall
        all_scores.append(scores)
        chances_by_model[model_name] = chances

    results_table = pd.DataFrame(all_scores).round(3)
    print("\n=== Model comparison ===")
    print(results_table.to_string(index=False))

    results_table.to_csv(RESULTS_FOLDER / "model_comparison.csv", index=False)
    save_roc_curve_picture(test_answers, chances_by_model)
    save_forest_importance_picture(
        trained_models["Random Forest"], patient_measurements.columns
    )
    print("\nSaved the table and pictures in the results folder.")


if __name__ == "__main__":
    main()