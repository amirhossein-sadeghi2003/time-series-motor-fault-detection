from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("data/processed/cwru_bearing_features.csv")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

TARGET_COLUMN = "label"
GROUP_COLUMN = "source_file"

DROP_COLUMNS = [
    "source_file",
    "source_folder",
    "label",
    "window_index",
    "start_index",
    "end_index",
]


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    feature_columns = [col for col in df.columns if col not in DROP_COLUMNS]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    groups = df[GROUP_COLUMN]

    return X, y, groups, feature_columns, df


def split_one_file_per_class(df):
    """
    Use a file-level split so windows from the same source file do not appear
    in both train and test.

    For each class, one source file is selected for the test set and the
    remaining source files stay in the training set. This keeps all classes
    represented in the test set.
    """
    test_files = []

    for label, group_df in df.groupby(TARGET_COLUMN):
        files = sorted(group_df[GROUP_COLUMN].unique())
        test_files.append(files[-1])

    test_mask = df[GROUP_COLUMN].isin(test_files)

    feature_columns = [col for col in df.columns if col not in DROP_COLUMNS]

    X_train = df.loc[~test_mask, feature_columns]
    X_test = df.loc[test_mask, feature_columns]
    y_train = df.loc[~test_mask, TARGET_COLUMN]
    y_test = df.loc[test_mask, TARGET_COLUMN]

    return X_train, X_test, y_train, y_test, test_files


def build_models():
    models = {
        "Decision Tree": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=8,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=5.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "MLP Neural Network": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        random_state=42,
                        early_stopping=False,
                    ),
                ),
            ]
        ),
    }

    return models


def safe_name(model_name):
    return model_name.lower().replace(" ", "_")


def save_confusion_matrix(model_name, model, X_test, y_test):
    output_path = RESULTS_DIR / f"confusion_matrix_{safe_name(model_name)}.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax,
        cmap="Blues",
        xticks_rotation=30,
    )
    ax.set_title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def save_model_comparison(results_df):
    output_path = RESULTS_DIR / "model_comparison.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(results_df["model"], results_df["macro_f1"])
    ax.set_ylabel("Macro F1-score")
    ax.set_title("Model Comparison on CWRU Bearing Fault Dataset")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def save_feature_importance(random_forest_pipeline, feature_columns):
    output_path = RESULTS_DIR / "feature_importance_random_forest.png"

    rf_model = random_forest_pipeline.named_steps["model"]
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    ).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups, feature_columns, df = load_dataset()
    X_train, X_test, y_train, y_test, test_files = split_one_file_per_class(df)

    print("Dataset shape:", X.shape)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print()
    print("Test source files:")
    for file_name in test_files:
        print("-", file_name)
    print()
    print("Train label distribution:")
    print(y_train.value_counts())
    print()
    print("Test label distribution:")
    print(y_test.value_counts())
    print()

    missing_values = X.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print("Columns with missing values:")
        print(missing_values)
        print("Missing values are handled using median imputation inside each model pipeline.")
        print()

    models = build_models()
    results = []

    for model_name, model in models.items():
        print("=" * 60)
        print(model_name)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print()
        print(classification_report(y_test, y_pred))

        confusion_path = save_confusion_matrix(model_name, model, X_test, y_test)
        print(f"Saved confusion matrix: {confusion_path}")

        model_path = MODELS_DIR / f"{safe_name(model_name)}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")
        print()

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )

    results_df = pd.DataFrame(results)
    results_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    comparison_plot_path = save_model_comparison(results_df)
    print(f"Saved model comparison plot: {comparison_plot_path}")

    feature_importance_path = save_feature_importance(models["Random Forest"], feature_columns)
    print(f"Saved feature importance plot: {feature_importance_path}")

    print()
    print("Model comparison:")
    print(results_df)


if __name__ == "__main__":
    main()
