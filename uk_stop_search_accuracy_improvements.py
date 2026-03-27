import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

RAW_ROOT = r"C:\Users\bazer\Desktop\PythonML\UK Stop and Search Data, February 2023 to January 2026"
MAX_FILES = 240
PER_FILE_SAMPLE = 350
RANDOM_STATE = 42


def load_sampled_raw_data() -> pd.DataFrame:
    all_raw_files = glob.glob(os.path.join(RAW_ROOT, "**/*.csv"), recursive=True)
    if not all_raw_files:
        raise FileNotFoundError(f"No CSV files found under: {RAW_ROOT}")

    rng = np.random.default_rng(RANDOM_STATE)
    if len(all_raw_files) > MAX_FILES:
        selected_idx = rng.choice(len(all_raw_files), size=MAX_FILES, replace=False)
        selected_files = [all_raw_files[i] for i in sorted(selected_idx)]
    else:
        selected_files = all_raw_files

    required_cols = [
        "Outcome",
        "Type",
        "Date",
        "Part of a policing operation",
        "Gender",
        "Age range",
        "Officer-defined ethnicity",
        "Legislation",
        "Object of search",
        "Outcome linked to object of search",
        "Removal of more than just outer clothing",
    ]

    frames = []
    for path in selected_files:
        try:
            df = pd.read_csv(path, low_memory=False)
            keep = [column for column in required_cols if column in df.columns]
            df = df[keep].copy()

            force_name = os.path.basename(path)
            force_name = re.sub(r"^\d{4}-\d{2}-", "", force_name)
            force_name = force_name.replace("-stop-and-search.csv", "")
            df["source_force"] = force_name

            if len(df) > PER_FILE_SAMPLE:
                df = df.sample(PER_FILE_SAMPLE, random_state=RANDOM_STATE)

            frames.append(df)
        except Exception:
            continue

    if not frames:
        raise ValueError("Could not load any raw data rows for experiments.")

    experiment_df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    experiment_df["Outcome"] = experiment_df["Outcome"].fillna("Unknown")
    experiment_df["Outcome_Binary"] = (
        experiment_df["Outcome"] != "A no further action disposal"
    ).astype(int)

    print(f"Files sampled: {len(selected_files)} / {len(all_raw_files)}")
    print(f"Experiment rows after dedupe: {len(experiment_df):,}")
    print("Class distribution:")
    print(experiment_df["Outcome_Binary"].value_counts(normalize=True).rename("ratio"))

    return experiment_df


def build_features(experiment_df: pd.DataFrame):
    experiment_df = experiment_df.copy()

    experiment_df["Date"] = pd.to_datetime(experiment_df.get("Date"), errors="coerce")
    experiment_df["year"] = experiment_df["Date"].dt.year.fillna(0).astype(int)
    experiment_df["month"] = experiment_df["Date"].dt.month.fillna(0).astype(int)
    experiment_df["day_of_week"] = experiment_df["Date"].dt.dayofweek.fillna(0).astype(int)
    experiment_df["hour"] = experiment_df["Date"].dt.hour.fillna(0).astype(int)
    experiment_df["is_weekend"] = experiment_df["day_of_week"].isin([5, 6]).astype(int)

    experiment_df["month_sin"] = np.sin(2 * np.pi * experiment_df["month"] / 12.0)
    experiment_df["month_cos"] = np.cos(2 * np.pi * experiment_df["month"] / 12.0)
    experiment_df["hour_sin"] = np.sin(2 * np.pi * experiment_df["hour"] / 24.0)
    experiment_df["hour_cos"] = np.cos(2 * np.pi * experiment_df["hour"] / 24.0)

    key_cat = [
        "Gender",
        "Age range",
        "Officer-defined ethnicity",
        "Legislation",
        "Object of search",
        "Type",
        "source_force",
    ]
    for column in key_cat:
        if column not in experiment_df.columns:
            experiment_df[column] = "Unknown"
        experiment_df[f"{column}_missing"] = experiment_df[column].isna().astype(int)
        experiment_df[column] = experiment_df[column].fillna("Unknown").astype(str)

    for column in [
        "Legislation",
        "Object of search",
        "source_force",
        "Officer-defined ethnicity",
    ]:
        value_counts = experiment_df[column].value_counts(normalize=True)
        rare_values = value_counts[value_counts < 0.01].index
        experiment_df[column] = experiment_df[column].where(
            ~experiment_df[column].isin(rare_values), "Other_Rare"
        )
        experiment_df[f"{column}_freq"] = (
            experiment_df[column]
            .map(experiment_df[column].value_counts(normalize=True))
            .astype(float)
        )

    experiment_df["gender_age_interaction"] = (
        experiment_df["Gender"] + "__" + experiment_df["Age range"]
    )
    experiment_df["type_object_interaction"] = (
        experiment_df["Type"] + "__" + experiment_df["Object of search"]
    )

    X = experiment_df.drop(columns=["Outcome", "Outcome_Binary", "Date"], errors="ignore")
    y = experiment_df["Outcome_Binary"].astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.2222,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    target_column = "Officer-defined ethnicity"
    train_target_map = y_train.groupby(X_train[target_column]).mean()
    global_mean = float(y_train.mean())

    for current_df in [X_train, X_val, X_test]:
        current_df[f"{target_column}_te"] = (
            current_df[target_column].map(train_target_map).fillna(global_mean)
        )

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_tuned_xgboost(X_train, X_val, X_test, y_train, y_val, y_test):
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [column for column in X_train.columns if column not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    ratio = float(negative_count / max(positive_count, 1))

    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 8],
        "gamma": [0, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.5, 1.0],
        "scale_pos_weight": [1.0, ratio, max(1.0, np.sqrt(ratio))],
    }

    best_auc = -1.0
    best_model = None
    best_params = None

    for trial_index, params in enumerate(
        ParameterSampler(param_grid, n_iter=18, random_state=RANDOM_STATE), start=1
    ):
        model = xgb.XGBClassifier(
            n_estimators=1200,
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=60,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **params,
        )
        model.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)
        validation_probabilities = model.predict_proba(X_val_t)[:, 1]
        validation_auc = roc_auc_score(y_val, validation_probabilities)

        print(f"Trial {trial_index:02d} | Val AUC: {validation_auc:.5f}")

        if validation_auc > best_auc:
            best_auc = validation_auc
            best_model = model
            best_params = params

    val_prob_best = best_model.predict_proba(X_val_t)[:, 1]
    test_prob_best = best_model.predict_proba(X_test_t)[:, 1]

    thresholds = np.linspace(0.10, 0.90, 161)
    best_threshold = float(
        max(
            thresholds,
            key=lambda threshold: accuracy_score(
                y_val, (val_prob_best >= threshold).astype(int)
            ),
        )
    )

    test_pred_best = (test_prob_best >= best_threshold).astype(int)

    xgb_metrics = {
        "threshold": best_threshold,
        "accuracy": accuracy_score(y_test, test_pred_best),
        "auc": roc_auc_score(y_test, test_prob_best),
        "precision": precision_score(y_test, test_pred_best, zero_division=0),
        "recall": recall_score(y_test, test_pred_best, zero_division=0),
        "f1": f1_score(y_test, test_pred_best, zero_division=0),
        "confusion": confusion_matrix(y_test, test_pred_best),
    }

    print("\nBest XGBoost settings:")
    print(best_params)

    return (
        xgb_metrics,
        thresholds,
        val_prob_best,
        test_prob_best,
        X_train_t,
        X_val_t,
        X_test_t,
    )


def train_ensemble(
    y_train,
    y_val,
    y_test,
    X_train_t,
    X_val_t,
    X_test_t,
    thresholds,
    val_prob_best,
    test_prob_best,
):
    lr_model = LogisticRegression(
        max_iter=1200,
        solver="saga",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    lr_model.fit(X_train_t, y_train)

    val_prob_lr = lr_model.predict_proba(X_val_t)[:, 1]
    test_prob_lr = lr_model.predict_proba(X_test_t)[:, 1]

    val_prob_ensemble = 0.75 * val_prob_best + 0.25 * val_prob_lr
    test_prob_ensemble = 0.75 * test_prob_best + 0.25 * test_prob_lr

    best_threshold_ensemble = float(
        max(
            thresholds,
            key=lambda threshold: accuracy_score(
                y_val, (val_prob_ensemble >= threshold).astype(int)
            ),
        )
    )

    test_pred_ensemble = (test_prob_ensemble >= best_threshold_ensemble).astype(int)

    ensemble_metrics = {
        "threshold": best_threshold_ensemble,
        "accuracy": accuracy_score(y_test, test_pred_ensemble),
        "auc": roc_auc_score(y_test, test_prob_ensemble),
        "precision": precision_score(y_test, test_pred_ensemble, zero_division=0),
        "recall": recall_score(y_test, test_pred_ensemble, zero_division=0),
        "f1": f1_score(y_test, test_pred_ensemble, zero_division=0),
        "confusion": confusion_matrix(y_test, test_pred_ensemble),
    }

    return ensemble_metrics


def print_metrics(label: str, metrics: dict):
    print(f"\n{label}")
    print(f"Threshold: {metrics['threshold']:.3f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print("Confusion matrix:")
    print(metrics["confusion"])


if __name__ == "__main__":
    raw_df = load_sampled_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = build_features(raw_df)

    (
        xgb_metrics,
        thresholds,
        val_prob_best,
        test_prob_best,
        X_train_t,
        X_val_t,
        X_test_t,
    ) = train_tuned_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)

    ensemble_metrics = train_ensemble(
        y_train,
        y_val,
        y_test,
        X_train_t,
        X_val_t,
        X_test_t,
        thresholds,
        val_prob_best,
        test_prob_best,
    )

    print_metrics("Tuned XGBoost", xgb_metrics)
    print_metrics("XGBoost + Logistic Ensemble", ensemble_metrics)
