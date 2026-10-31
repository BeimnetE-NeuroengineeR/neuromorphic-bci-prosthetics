
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from mne.decoding import CSP

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/A01T.mat"
RESULTS = ROOT / "results/metrics"
FIGURES = ROOT / "results/figures"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

FS = 250
CLASS_NAMES = ["Left hand", "Right hand", "Feet", "Tongue"]


def load_trials():
    runs = loadmat(DATA, simplify_cells=True)["data"]

    signals, labels = [], []

    sos = butter(
        4, [8, 30],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    for run in runs:
        y = np.asarray(run["y"]).reshape(-1)

        if len(y) == 0:
            continue

        eeg = np.asarray(run["X"], dtype=np.float32)[:, :22]
        eeg = sosfiltfilt(sos, eeg, axis=0)

        trials = np.asarray(run["trial"]).reshape(-1)
        artifacts = np.asarray(run["artifacts"]).reshape(-1)

        for i, (position, label) in enumerate(zip(trials, y)):
            if i < len(artifacts) and artifacts[i] != 0:
                continue

            # Trial start + 3 seconds; extract 3-second MI period
            start = int(position) - 1 + 3 * FS
            end = start + 3 * FS

            if start < 0 or end > len(eeg):
                continue

            signals.append(eeg[start:end].T)
            labels.append(int(label) - 1)

    return np.asarray(signals), np.asarray(labels)


def main():
    print("Loading EEG data...")

    X, y = load_trials()

    print("Trials:", len(y))
    print("EEG shape:", X.shape)
    print("Class counts:", np.bincount(y, minlength=4))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\nFitting CSP features...")

    csp = CSP(
        n_components=6,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False
    )

    X_train_csp = csp.fit_transform(X_train, y_train)
    X_test_csp = csp.transform(X_test)

    print("CSP training features:", X_train_csp.shape)

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X_train_csp, y_train)

    predictions = classifier.predict(X_test_csp)

    accuracy = accuracy_score(y_test, predictions)

    print("\nCSP + LDA TEST RESULTS")
    print("Accuracy:", round(accuracy, 4))

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    pd.DataFrame({
        "true_label": [CLASS_NAMES[i] for i in y_test],
        "predicted_label": [CLASS_NAMES[i] for i in predictions]
    }).to_csv(
        RESULTS / "csp_lda_predictions.csv",
        index=False
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2, 3]
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")

    ax.set(
        xticks=np.arange(4),
        yticks=np.arange(4),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="CSP + LDA Confusion Matrix"
    )

    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(
        FIGURES / "csp_lda_confusion_matrix.png",
        dpi=300
    )

    plt.close(fig)

    with open(RESULTS / "csp_lda_performance.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Training trials: {len(y_train)}\n")
        f.write(f"Testing trials: {len(y_test)}\n")

    print("\nBaseline results saved.")


if __name__ == "__main__":
    main()
