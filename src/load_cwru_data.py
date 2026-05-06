from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


RAW_DATA_DIR = Path("data/raw/cwru")
PROCESSED_DATA_DIR = Path("data/processed")

WINDOW_SIZE = 1024
STEP_SIZE = 512


LABEL_MAP = {
    "normal": "normal",
    "ball_007": "ball_fault",
    "inner_race_007": "inner_race_fault",
    "outer_race_007": "outer_race_fault",
}


def find_de_signal_key(mat_dict):
    for key in mat_dict.keys():
        if key.endswith("_DE_time"):
            return key
    raise KeyError("No drive-end vibration key ending with '_DE_time' found.")


def find_rpm_key(mat_dict):
    for key in mat_dict.keys():
        if key.endswith("RPM"):
            return key
    return None


def extract_windows(signal, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    windows = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        windows.append((start, end, signal[start:end]))

    return windows


def load_single_mat_file(file_path, label, source_folder):
    mat = loadmat(file_path)

    de_key = find_de_signal_key(mat)
    rpm_key = find_rpm_key(mat)

    signal = mat[de_key].squeeze().astype(float)

    rpm = np.nan
    if rpm_key is not None:
        rpm_value = mat[rpm_key].squeeze()
        if np.size(rpm_value) > 0:
            rpm = float(np.ravel(rpm_value)[0])

    rows = []
    windows = extract_windows(signal)

    for window_index, (start, end, window) in enumerate(windows):
        rows.append(
            {
                "source_file": file_path.name,
                "source_folder": source_folder,
                "label": label,
                "window_index": window_index,
                "start_index": start,
                "end_index": end,
                "rpm": rpm,
                "signal_mean": np.mean(window),
                "signal_std": np.std(window),
                "signal_min": np.min(window),
                "signal_max": np.max(window),
                "signal_range": np.max(window) - np.min(window),
                "signal_rms": np.sqrt(np.mean(window**2)),
                "signal_energy": np.sum(window**2),
                "signal_abs_mean": np.mean(np.abs(window)),
                "signal_peak_abs": np.max(np.abs(window)),
                "signal_crest_factor": np.max(np.abs(window)) / (np.sqrt(np.mean(window**2)) + 1e-12),
                "signal_skew_like": np.mean(((window - np.mean(window)) / (np.std(window) + 1e-12)) ** 3),
                "signal_kurtosis_like": np.mean(((window - np.mean(window)) / (np.std(window) + 1e-12)) ** 4),
            }
        )

    return rows


def build_processed_dataset():
    all_rows = []

    for folder_name, label in LABEL_MAP.items():
        folder_path = RAW_DATA_DIR / folder_name

        if not folder_path.exists():
            raise FileNotFoundError(f"Missing folder: {folder_path}")

        mat_files = sorted(folder_path.glob("*.mat"))

        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {folder_path}")

        for file_path in mat_files:
            rows = load_single_mat_file(
                file_path=file_path,
                label=label,
                source_folder=folder_name,
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    return df


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = build_processed_dataset()

    output_path = PROCESSED_DATA_DIR / "cwru_bearing_features.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved processed dataset to: {output_path}")
    print(f"Shape: {df.shape}")
    print()
    print("Label distribution:")
    print(df["label"].value_counts())
    print()
    print("Source files:")
    print(df.groupby(["label", "source_file"]).size())


if __name__ == "__main__":
    main()
