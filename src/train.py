"""Train models with MLflow tracking + Model Registry."""
import os
import sys
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report, precision_recall_curve)
from sklearn.pipeline import Pipeline
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import build_preprocessor, load_and_split

mlflow.set_experiment("Heart_Disease_Prediction_Pronab")
REGISTERED_MODEL_NAME = "HeartDiseasePredictor_Pronab"


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - {model_name}", fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path = f"confusion_matrix_{model_name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", linewidth=2, color="#e74c3c")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    path = f"roc_{model_name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def plot_pr_curve(y_true, y_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2, color="#3498db")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}", fontweight="bold")
    plt.grid(alpha=0.3)
    path = f"pr_curve_{model_name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def save_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, target_names=["No Disease", "Disease"])
    path = f"classification_report_{model_name}.txt"
    with open(path, "w") as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    return path


def train_and_log(model, param_grid, model_name, X_train, X_test, y_train, y_test, preprocessor):
    with mlflow.start_run(run_name=model_name) as run:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            pipeline, param_grid,
            cv=cv_strategy, scoring="roc_auc",
            n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(best_model, X_train, y_train,
                                     cv=cv_strategy, scoring="roc_auc")

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "cv_roc_auc_mean": cv_scores.mean(),
            "cv_roc_auc_std": cv_scores.std(),
        }

        mlflow.log_params(grid.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("model_type", model_name)
        mlflow.log_metrics(metrics)

        mlflow.set_tag("author", "Pronab Sardar")
        mlflow.set_tag("dataset", "UCI Heart Disease")
        mlflow.set_tag("stage", "development")

        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        roc_path = plot_roc_curve(y_test, y_proba, model_name)
        pr_path = plot_pr_curve(y_test, y_proba, model_name)
        report_path = save_classification_report(y_test, y_pred, model_name)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)
        mlflow.log_artifact(report_path)

        for p in [cm_path, roc_path, pr_path, report_path]:
            if os.path.exists(p):
                os.remove(p)

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=f"{REGISTERED_MODEL_NAME}_{model_name}"
        )

        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        print(f"Best params: {grid.best_params_}")
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"Run ID: {run.info.run_id}")

        return best_model, metrics, run.info.run_id


def promote_to_production(run_id, model_name):
    client = MlflowClient()
    try:
        registered_name = f"{REGISTERED_MODEL_NAME}_{model_name}"
        versions = client.search_model_versions(f"name='{registered_name}'")
        if versions:
            latest_version = max(int(v.version) for v in versions)
            client.transition_model_version_stage(
                name=registered_name,
                version=latest_version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"\nModel {registered_name} v{latest_version} promoted to PRODUCTION")
    except Exception as e:
        print(f"Registry promotion warning: {e}")


def main():
    print("Starting Heart Disease Model Training Pipeline...")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    preprocessor = build_preprocessor()

    models_config = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs"]
            }
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_split": [2, 5]
            }
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 6],
                "classifier__learning_rate": [0.05, 0.1]
            }
        ),
    }

    results = {}
    for name, (model, params) in models_config.items():
        best_model, metrics, run_id = train_and_log(
            model, params, name, X_train, X_test, y_train, y_test, preprocessor
        )
        results[name] = {"model": best_model, "metrics": metrics, "run_id": run_id}

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    comparison_df = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T
    print(comparison_df.round(4))

    os.makedirs("models", exist_ok=True)
    comparison_df.to_csv("models/model_comparison.csv")

    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best_model = results[best_name]["model"]
    best_run_id = results[best_name]["run_id"]

    joblib.dump(best_model, "models/best_model.pkl")
    promote_to_production(best_run_id, best_name)

    print(f"\nBEST MODEL: {best_name}")
    print(f"   ROC-AUC: {results[best_name]['metrics']['roc_auc']:.4f}")
    print(f"   Saved to: models/best_model.pkl")


if __name__ == "__main__":
    main()