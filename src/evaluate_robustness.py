from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from train_models import build_models, split_one_file_per_class


DATA_PATH = Path("data/processed/cwru_bearing_features.csv")
RESULTS_DIR = Path("results")

TARGET_COLUMN = "label"

DROP_COLUMNS = [
    "source_file",
    "source_folder",
    "label",
    "window_index",
    "start_index",
    "end_index",
]

NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30]
N_REPEATS = 5
RANDOM_SEED = 42


def safe_name(model_name):
    return model_name.lower().replace(" ", "_")


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def add_feature_noise(X, feature_stds, noise_level, rng):
    """
    Add Gaussian noise to engineered feature values.

    The noise standard deviation for each feature is proportional to the
    training-set standard deviation of that feature:

        noise_std(feature) = noise_level * train_std(feature)

    Missing values are preserved as NaN so that each model pipeline can handle
    them using its existing imputation step.
    """
    if noise_level == 0.0:
        return X.copy()

    X_noisy = X.copy()

    noise = rng.normal(
        loc=0.0,
        scale=noise_level,
        size=X_noisy.shape,
    )

    noise_df = pd.DataFrame(
        noise,
        index=X_noisy.index,
        columns=X_noisy.columns,
    )

    scaled_noise = noise_df.multiply(feature_stds, axis=1)

    X_noisy = X_noisy + scaled_noise

    return X_noisy


def run_robustness_experiment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    X_train, X_test, y_train, y_test, test_files = split_one_file_per_class(df)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print()
    print("Test source files:")
    for file_name in test_files:
        print("-", file_name)
    print()

    feature_stds = X_train.std(axis=0, skipna=True)
    feature_stds = feature_stds.replace(0, 1e-12)

    models = build_models()

    trained_models = {}

    print("Training models on clean training features...")
    print()

    for model_name, model in models.items():
        print(f"Training: {model_name}")
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    print()
    print("Evaluating robustness under feature noise...")
    print()

    rows = []

    for model_name, model in trained_models.items():
        print("=" * 60)
        print(model_name)

        for noise_level in NOISE_LEVELS:
            repeat_accuracies = []
            repeat_macro_f1s = []

            for repeat_idx in range(N_REPEATS):
                rng = np.random.default_rng(RANDOM_SEED + repeat_idx)

                X_test_noisy = add_feature_noise(
                    X=X_test,
                    feature_stds=feature_stds,
                    noise_level=noise_level,
                    rng=rng,
                )

                y_pred = model.predict(X_test_noisy)

                accuracy = accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average="macro")

                repeat_accuracies.append(accuracy)
                repeat_macro_f1s.append(macro_f1)

            mean_accuracy = float(np.mean(repeat_accuracies))
            std_accuracy = float(np.std(repeat_accuracies))

            mean_macro_f1 = float(np.mean(repeat_macro_f1s))
            std_macro_f1 = float(np.std(repeat_macro_f1s))

            print(
                f"noise={noise_level:.2f} | "
                f"accuracy={mean_accuracy:.4f} ± {std_accuracy:.4f} | "
                f"macro_f1={mean_macro_f1:.4f} ± {std_macro_f1:.4f}"
            )

            rows.append(
                {
                    "model": model_name,
                    "noise_level": noise_level,
                    "accuracy_mean": mean_accuracy,
                    "accuracy_std": std_accuracy,
                    "macro_f1_mean": mean_macro_f1,
                    "macro_f1_std": std_macro_f1,
                    "n_repeats": N_REPEATS,
                }
            )

        print()

    results_df = pd.DataFrame(rows)

    output_csv = RESULTS_DIR / "robustness_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Saved robustness results to: {output_csv}")

    return results_df

def plot_metric(results_df, metric_mean, metric_std, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    style_map = {
        "Decision Tree": {"marker": "o", "linestyle": "-"},
        "Random Forest": {"marker": "s", "linestyle": "--"},
        "SVM": {"marker": "^", "linestyle": "-."},
        "MLP Neural Network": {"marker": "D", "linestyle": ":"},
    }

    for model_name, model_df in results_df.groupby("model"):
        model_df = model_df.sort_values("noise_level")
        style = style_map.get(model_name, {"marker": "o", "linestyle": "-"})

        ax.errorbar(
            model_df["noise_level"],
            model_df[metric_mean],
            yerr=model_df[metric_std],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=6,
            capsize=3,
            label=model_name,
        )

    ax.set_xlabel("Feature noise level")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.70, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved plot to: {output_path}")


def main():
    results_df = run_robustness_experiment()

    plot_metric(
        results_df=results_df,
        metric_mean="accuracy_mean",
        metric_std="accuracy_std",
        ylabel="Accuracy",
        title="Robustness to Feature Noise: Accuracy",
        output_path=RESULTS_DIR / "robustness_accuracy_vs_noise.png",
    )

    plot_metric(
        results_df=results_df,
        metric_mean="macro_f1_mean",
        metric_std="macro_f1_std",
        ylabel="Macro F1-score",
        title="Robustness to Feature Noise: Macro F1-score",
        output_path=RESULTS_DIR / "robustness_macro_f1_vs_noise.png",
    )


if __name__ == "__main__":
    main()
