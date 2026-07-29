"""
Generic hybrid (ML + DNN) trainer, driven by config/diseases.json.

Usage:
    python train.py                  # trains every configured condition
    python train.py diabetes         # trains just one condition
    python train.py diabetes heart_disease
"""
import os
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from dl_model import build_dnn
from config_loader import load_disease_config

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def get_ml_models():
    """Each pipeline oversamples the minority class (SMOTE) BEFORE fitting the
    classifier. imblearn's Pipeline only resamples during .fit() — .predict()
    and .predict_proba() pass through untouched, so validation/test data and
    every cross-validation fold's held-out portion are never resampled,
    only the data the model actually trains on."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("resample", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "decision_tree": Pipeline([
            ("resample", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))
        ]),
        "random_forest": Pipeline([
            ("resample", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))
        ]),
        "gradient_boosting": Pipeline([
            ("resample", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ])
    }


def cross_validate_ml_models(X_trainval, y_trainval):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in get_ml_models().items():
        f1_scores = cross_val_score(model, X_trainval, y_trainval, cv=cv, scoring="f1")
        auc_scores = cross_val_score(model, X_trainval, y_trainval, cv=cv, scoring="roc_auc")
        rows.append({
            "model": name,
            "f1_mean": f1_scores.mean(), "f1_std": f1_scores.std(),
            "roc_auc_mean": auc_scores.mean(), "roc_auc_std": auc_scores.std(),
        })
        print(f"  {name}: F1 = {f1_scores.mean():.4f} +/- {f1_scores.std():.4f}, "
              f"ROC-AUC = {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")
    return pd.DataFrame(rows).set_index("model")


def evaluate_model(name, y_test, y_pred, y_prob):
    results = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    if y_prob is not None:
        results["roc_auc"] = roc_auc_score(y_test, y_prob)
    print(f"\n  {name}: {results}")
    print(classification_report(y_test, y_pred))
    return results


def train_dnn(X_train, y_train, X_val, y_val, scaler):
    """Fits on SMOTE-oversampled training data; scaler and validation set
    stay on the real (unresampled) distribution."""
    X_train_res, y_train_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    dnn = build_dnn(input_dim=X_train.shape[1])
    dnn.fit(
        scaler.transform(X_train_res), y_train_res,
        validation_data=(scaler.transform(X_val), y_val),
        epochs=100, batch_size=16, verbose=0
    )
    return dnn


def plot_roc_curves(roc_data, path, title):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, name, path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison_table(results_df, path, title):
    fig, ax = plt.subplots(figsize=(9.5, 1.2 + 0.4 * len(results_df)))
    ax.axis("off")
    tbl = ax.table(
        cellText=np.round(results_df.values, 4),
        colLabels=results_df.columns,
        rowLabels=results_df.index,
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def train_condition(disease_key, disease_cfg):
    print(f"\n{'=' * 60}\nTraining: {disease_cfg['display_name']} ({disease_key})\n{'=' * 60}")

    model_dir = os.path.join("models", disease_key)
    report_dir = os.path.join("reports", disease_key)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    df = pd.read_csv(disease_cfg["data_path"])
    target_col = disease_cfg["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("\n5-fold cross-validation (train+val pool):")
    cv_df = cross_validate_ml_models(X_trainval, y_trainval)
    cv_df.to_csv(os.path.join(report_dir, "cv_comparison.csv"))
    plot_comparison_table(cv_df, os.path.join(report_dir, "cv_comparison.png"),
                           f"{disease_cfg['display_name']} - 5-Fold CV")

    best_ml_name = cv_df["f1_mean"].idxmax()
    print(f"\nBest ML model by CV F1: {best_ml_name}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=RANDOM_STATE, stratify=y_trainval
    )

    best_ml_model = get_ml_models()[best_ml_name]
    best_ml_model.fit(X_train, y_train)
    ml_val_prob = best_ml_model.predict_proba(X_val)[:, 1]
    ml_val_f1 = f1_score(y_val, (ml_val_prob >= 0.5).astype(int))

    scaler = StandardScaler().fit(X_train)
    dnn = train_dnn(X_train, y_train, X_val, y_val, scaler)
    dnn_val_prob = dnn.predict(scaler.transform(X_val), verbose=0).ravel()
    dnn_val_f1 = f1_score(y_val, (dnn_val_prob >= 0.5).astype(int))

    print(f"Validation F1 -> ML ({best_ml_name}): {ml_val_f1:.4f} | DNN: {dnn_val_f1:.4f}")

    total = ml_val_f1 + dnn_val_f1
    w_ml, w_dnn = (0.5, 0.5) if total == 0 else (ml_val_f1 / total, dnn_val_f1 / total)
    print(f"Hybrid weights -> ML: {w_ml:.3f}, DNN: {w_dnn:.3f}")

    final_ml_model = get_ml_models()[best_ml_name]
    final_ml_model.fit(X_trainval, y_trainval)

    final_scaler = StandardScaler().fit(X_trainval)
    X_trainval_res, y_trainval_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_trainval, y_trainval)
    final_dnn = build_dnn(input_dim=X_trainval.shape[1])
    final_dnn.fit(final_scaler.transform(X_trainval_res), y_trainval_res, epochs=100, batch_size=16, verbose=0)

    all_results = []
    roc_data = {}

    ml_test_prob = final_ml_model.predict_proba(X_test)[:, 1]
    ml_test_pred = (ml_test_prob >= 0.5).astype(int)
    result = evaluate_model(best_ml_name, y_test, ml_test_pred, ml_test_prob)
    all_results.append(result)
    fpr, tpr, _ = roc_curve(y_test, ml_test_prob)
    roc_data[best_ml_name] = (fpr, tpr, result["roc_auc"])

    dnn_test_prob = final_dnn.predict(final_scaler.transform(X_test), verbose=0).ravel()
    dnn_test_pred = (dnn_test_prob >= 0.5).astype(int)
    dnn_result = evaluate_model("dnn", y_test, dnn_test_pred, dnn_test_prob)
    all_results.append(dnn_result)
    fpr, tpr, _ = roc_curve(y_test, dnn_test_prob)
    roc_data["dnn"] = (fpr, tpr, dnn_result["roc_auc"])

    hybrid_prob = w_ml * ml_test_prob + w_dnn * dnn_test_prob
    hybrid_pred = (hybrid_prob >= 0.5).astype(int)
    hybrid_result = evaluate_model(f"hybrid (weighted {w_ml:.2f}/{w_dnn:.2f})",
                                    y_test, hybrid_pred, hybrid_prob)
    all_results.append(hybrid_result)
    fpr, tpr, _ = roc_curve(y_test, hybrid_prob)
    roc_data["hybrid"] = (fpr, tpr, hybrid_result["roc_auc"])

    results_df = pd.DataFrame(all_results).set_index("model")
    results_df.to_csv(os.path.join(report_dir, "test_comparison.csv"))
    plot_comparison_table(results_df, os.path.join(report_dir, "test_comparison.png"),
                           f"{disease_cfg['display_name']} - Held-out Test Performance")
    plot_roc_curves(roc_data, os.path.join(report_dir, "roc_curves.png"),
                     f"{disease_cfg['display_name']} - ROC Curves")
    plot_confusion_matrix(y_test, hybrid_pred, "Weighted Hybrid",
                           os.path.join(report_dir, "confusion_matrix_hybrid.png"))

    joblib.dump(final_ml_model, os.path.join(model_dir, "best_ml_model.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, "feature_columns.pkl"))
    joblib.dump(final_scaler, os.path.join(model_dir, "scaler.pkl"))
    final_dnn.save(os.path.join(model_dir, "dnn_model.keras"))

    background = X_trainval.sample(min(100, len(X_trainval)), random_state=RANDOM_STATE)
    joblib.dump(background, os.path.join(model_dir, "shap_background.pkl"))

    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump({
            "best_ml_model": best_ml_name,
            "hybrid_weights": {"ml": w_ml, "dnn": w_dnn},
            "validation_f1": {"ml": ml_val_f1, "dnn": dnn_val_f1}
        }, f, indent=2)

    print(f"\nSaved models -> {model_dir}/, reports -> {report_dir}/")
    return cv_df, results_df


def main():
    config = load_disease_config()
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(config.keys())

    for disease_key in requested:
        if disease_key not in config:
            print(f"Skipping unknown condition: {disease_key}")
            continue
        train_condition(disease_key, config[disease_key])


if __name__ == "__main__":
    main()
