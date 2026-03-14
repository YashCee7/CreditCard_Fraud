#!/usr/bin/env python3
"""
creditcard_fraud_detection.py

Complete script to train and evaluate multiple classifiers on the credit card fraud dataset
with several resampling strategies. Outputs confusion matrices, AUC scores, and ROC plots.

Usage:
    python creditcard_fraud_detection.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
    accuracy_score,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 0


# Try imports that may not be present; give actionable error messages.
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    # imblearn provides the resampling techniques seen in the screenshots
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks
    from imblearn.combine import SMOTETomek
except Exception:
    RandomOverSampler = SMOTE = RandomUnderSampler = ClusterCentroids = TomekLinks = SMOTETomek = None


def check_dependencies():
    missing = []
    if XGBClassifier is None:
        missing.append("xgboost (pip install xgboost)")
    if RandomOverSampler is None:
        missing.append("imbalanced-learn (pip install imbalanced-learn)")
    if missing:
        print("Missing optional dependencies required for full functionality:")
        for m in missing:
            print("  -", m)
        print("\nInstall them to enable XGBoost and resampling options. The script will continue with what's available.\n")


def load_data(csv_path="creditcard.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' not found. Put dataset in the same folder or provide correct path.")
    df = pd.read_csv(csv_path)
    if "Class" not in df.columns:
        raise ValueError("CSV must contain a 'Class' column (0 = normal, 1 = fraud).")
    return df


def preprocess(df):
    """
    Create X, y. We'll use all features except 'Class' as X.
    The commonly used dataset has columns: Time, V1..V28, Amount, Class.
    We will drop 'Time' because it's of less value in many experiments (optionally you can keep it).
    """
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])  # optional: drop 'Time'
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    return X, y


def get_samplers():
    """
    Return dict of sampler_name: sampler_instance_or_None.
    None means 'no resampling' (use original distribution).
    """
    samplers = {"Normal": None}

    if RandomOverSampler is not None:
        samplers["Random Oversample"] = RandomOverSampler(random_state=RANDOM_STATE)
    if RandomUnderSampler is not None:
        samplers["Random Undersample"] = RandomUnderSampler(random_state=RANDOM_STATE)
    if ClusterCentroids is not None:
        samplers["Cluster Centroids"] = ClusterCentroids(random_state=RANDOM_STATE)
    if TomekLinks is not None:
        samplers["Tomek Links (undersample)"] = TomekLinks()
    if SMOTE is not None:
        samplers["SMOTE"] = SMOTE(random_state=RANDOM_STATE)
    if SMOTETomek is not None:
        samplers["SMOTE+Tomek"] = SMOTETomek(random_state=RANDOM_STATE)

    return samplers


def get_classifiers():
    """
    Return dict of short_name: classifier_instance
    Hyperparameters are taken from the screenshots or sensible defaults.
    """
    clfs = {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
        "DecisionTree": DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_leaf=6, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=30, oob_score=False, random_state=RANDOM_STATE),
    }
    if XGBClassifier is not None:
        # avoid use_label_encoder warning in newer xgboost
        clfs["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
    return clfs


def fit_and_evaluate(clf, X_train, y_train, X_test, y_test, plot_roc=False, plot_label=None):
    """
    Fit classifier, return dict with confusion matrix and AUC and other metrics.
    If classifier doesn't implement predict_proba, try decision_function or fallback.
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # obtain positive-class probabilities
    probs = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        # decision_function sometimes gives raw scores
        scores = clf.decision_function(X_test)
        # convert scores to a 0..1-ish range using a logistic - but roc_curve only needs relative ordering;
        # use a stable mapping: use scores directly
        probs = scores
    else:
        # fallback: use predictions (discrete) - not ideal for ROC/AUC but avoid crash
        probs = y_pred
,
    cm = confusion_matrix(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = float("nan")

    metrics = {
        "confusion_matrix": cm,
        "auc": auc,
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=4),
    }

    if plot_roc:
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, marker=".", label=plot_label or "ROC")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True)

    return metrics


def experiment(df, csv_out="results_summary.csv", do_plot=True):
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    samplers = get_samplers()
    clfs = get_classifiers()

    # collect results in a DataFrame
    results = []

    # for ROC plotting per classifier across sampling strategies: create separate figures per classifier
    for clf_name, clf in clfs.items():
        if do_plot:
            plt.figure(figsize=(7, 6))
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
            plt.title(f"ROC curves for {clf_name}")
        for sampler_name, sampler in samplers.items():
            # prepare training data (resample if sampler is present)
            if sampler is None:
                Xr_train, yr_train = X_train, y_train
            else:
                try:
                    Xr_train, yr_train = sampler.fit_resample(X_train, y_train)
                except Exception as e:
                    print(f"Sampler {sampler_name} failed: {e}. Skipping.")
                    continue

            metrics = fit_and_evaluate(
                clf,
                Xr_train,
                yr_train,
                X_test,
                y_test,
                plot_roc=do_plot,
                plot_label=f"{sampler_name} (AUC ?)",
            )

            # because we plotted without updating label with AUC, re-plot with label including AUC:
            if do_plot:
                # recompute probs for this exact fitted model to get ROC for label
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_test)[:, 1]
                elif hasattr(clf, "decision_function"):
                    probs = clf.decision_function(X_test)
                else:
                    probs = clf.predict(X_test)
                fpr, tpr, _ = roc_curve(y_test, probs)
                plt.plot(fpr, tpr, marker=".", label=f"{sampler_name} (AUC {roc_auc_score(y_test, probs):.3f})")

            # store results
            results.append(
                {
                    "classifier": clf_name,
                    "sampler": sampler_name,
                    "auc": metrics["auc"],
                    "accuracy": metrics["accuracy"],
                    "tp": int(metrics["confusion_matrix"][1, 1]) if metrics["confusion_matrix"].shape == (2, 2) else 0,
                    "tn": int(metrics["confusion_matrix"][0, 0]) if metrics["confusion_matrix"].shape == (2, 2) else 0,
                    "fp": int(metrics["confusion_matrix"][0, 1]) if metrics["confusion_matrix"].shape == (2, 2) else 0,
                    "fn": int(metrics["confusion_matrix"][1, 0]) if metrics["confusion_matrix"].shape == (2, 2) else 0,
                    "classification_report": metrics["classification_report"],
                }
            )

        if do_plot:
            plt.legend(loc="lower right", fontsize="small")
            plt.tight_layout()
            plt.show()

    results_df = pd.DataFrame(results)
    # reorder and format
    results_df = results_df[["classifier", "sampler", "auc", "accuracy", "tp", "tn", "fp", "fn", "classification_report"]]
    results_df = results_df.sort_values(["classifier", "sampler"]).reset_index(drop=True)
    results_df.to_csv(csv_out, index=False)
    return results_df


def print_summary_table(results_df):
    """
    Nicely print a compact summary: per classifier print sampler and AUC.
    """
    classifiers = results_df["classifier"].unique()
    for clf in classifiers:
        sub = results_df[results_df["classifier"] == clf].copy()
        print(f"\nAccuracy comparison for {clf}")
        print("-" * 40)
        for _, row in sub.iterrows():
            auc = row["auc"]
            auc_str = f"{auc:.3f}" if pd.notna(auc) else "nan"
            print(f"{row['sampler']:<20} | {auc_str:>6}")
    print("\nDetailed results CSV saved as 'results_summary.csv'.")


def main():
    check_dependencies()
    try:
        df = load_data("creditcard.csv")
    except Exception as e:
        print("Error loading data:", e)
        sys.exit(1)

    print("Dataset loaded. Class distribution:")
    print(df["Class"].value_counts(normalize=False))
    print(df["Class"].value_counts(normalize=True))

    results_df = experiment(df, csv_out="results_summary.csv", do_plot=True)
    print_summary_table(results_df)


if __name__ == "__main__":
    main()