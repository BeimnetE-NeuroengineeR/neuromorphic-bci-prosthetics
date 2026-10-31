
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, resample

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from mne.decoding import CSP

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
FS = 250
N_CHANNELS = 22
WINDOW_SECONDS = 3

N_CSP_COMPONENTS = 6
STEPS = 100

HIDDEN_SIZE = 64
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DATA_PATH = "data/raw/A01T.mat"

CLASS_NAMES = [
    "Left hand",
    "Right hand",
    "Feet",
    "Tongue",
]

OUTPUT_DIR = "results/runwise_validation"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)


# ============================================================
# 2. LOAD DATA AND RECORDING-RUN IDENTIFIERS
# ============================================================

def load_trials():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    print("\nLoading EEG data...")

    data = loadmat(
        DATA_PATH,
        simplify_cells=True
    )["data"]

    signals = []
    labels = []
    run_ids = []

    sos = butter(
        4,
        [8, 30],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    for run_index, run in enumerate(data):

        y = np.asarray(run["y"]).reshape(-1)
        positions = np.asarray(run["trial"]).reshape(-1)
        artifacts = np.asarray(run["artifacts"]).reshape(-1)

        if len(y) == 0:
            continue

        eeg = np.asarray(
            run["X"],
            dtype=np.float64
        )[:, :N_CHANNELS]

        eeg = sosfiltfilt(
            sos,
            eeg,
            axis=0
        )

        for position, label, artifact in zip(
            positions,
            y,
            artifacts
        ):

            if artifact != 0:
                continue

            start = int(position) - 1 + 3 * FS
            end = start + WINDOW_SECONDS * FS

            if end > len(eeg):
                continue

            trial = eeg[start:end].T

            if trial.shape != (
                N_CHANNELS,
                WINDOW_SECONDS * FS
            ):
                continue

            signals.append(trial)
            labels.append(int(label) - 1)
            run_ids.append(run_index)

    X = np.asarray(signals, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    run_ids = np.asarray(run_ids)

    print("Total clean trials:", len(X))
    print("EEG shape:", X.shape)
    print("Class counts:", np.bincount(y))
    print("Recording runs:", np.unique(run_ids))

    return X, y, run_ids


# ============================================================
# 3. HYBRID CSP-SNN MODEL
# ============================================================

class HybridCSPSNN(nn.Module):

    def __init__(
        self,
        input_size=N_CSP_COMPONENTS * 2,
        hidden_size=HIDDEN_SIZE,
        num_classes=4
    ):

        super().__init__()

        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(
            input_size,
            hidden_size
        )

        self.recurrent_layer = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.lif = snn.Leaky(
            beta=0.9,
            spike_grad=surrogate.fast_sigmoid(
                slope=25
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_activity=False):

        batch_size, time_steps, _ = x.shape

        mem = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device
        )

        spk = torch.zeros_like(mem)

        spike_history = []
        membrane_history = []

        for t in range(time_steps):

            current = (
                self.input_layer(x[:, t])
                + self.recurrent_layer(spk)
            )

            spk, mem = self.lif(current, mem)

            spike_history.append(spk)
            membrane_history.append(mem)

        hidden_spikes = torch.stack(
            spike_history,
            dim=1
        )

        hidden_membranes = torch.stack(
            membrane_history,
            dim=1
        )

        spike_rates = hidden_spikes.mean(dim=1)
        membrane_activity = hidden_membranes.mean(dim=1)

        features = torch.cat(
            [spike_rates, membrane_activity],
            dim=1
        )

        logits = self.classifier(features)

        if return_activity:
            return logits, hidden_spikes

        return logits


# ============================================================
# 4. TRAINING
# ============================================================

def train_snn(model, X_train, y_train):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6
    )

    for epoch in range(EPOCHS):

        model.train()

        permutation = torch.randperm(
            len(X_train),
            device=device
        )

        total_loss = 0.0

        for start in range(0, len(X_train), BATCH_SIZE):

            indices = permutation[
                start:start + BATCH_SIZE
            ]

            batch_x = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(indices)

        average_loss = total_loss / len(X_train)
        scheduler.step(average_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"    Epoch {epoch + 1}/{EPOCHS}"
                f" | Loss: {average_loss:.4f}"
            )


# ============================================================
# 5. HYBRID MODEL PREPROCESSING
# ============================================================

def prepare_snn_data(X_train, X_test):

    # Fit CSP only on the training runs
    csp = CSP(
        n_components=N_CSP_COMPONENTS,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False
    )

    csp.fit(X_train, y_train_fold)

    filters = csp.filters_[:N_CSP_COMPONENTS]

    X_train_csp = np.einsum(
        "kc,nct->nkt",
        filters,
        X_train
    )

    X_test_csp = np.einsum(
        "kc,nct->nkt",
        filters,
        X_test
    )

    # Fit normalization only on training data
    scaler = StandardScaler()

    train_flat = X_train_csp.transpose(
        0, 2, 1
    ).reshape(-1, N_CSP_COMPONENTS)

    scaler.fit(train_flat)

    def scale_components(X):

        n_trials, n_components, n_times = X.shape

        flat = X.transpose(
            0, 2, 1
        ).reshape(-1, n_components)

        transformed = scaler.transform(flat)

        return transformed.reshape(
            n_trials,
            n_times,
            n_components
        ).transpose(0, 2, 1)

    X_train_scaled = scale_components(X_train_csp)
    X_test_scaled = scale_components(X_test_csp)

    # Learn thresholds using training data only
    sampled_train = np.asarray([
        resample(trial, STEPS, axis=1)
        for trial in X_train_scaled
    ])

    train_delta = np.diff(
        sampled_train,
        axis=2,
        prepend=sampled_train[:, :, :1]
    )

    thresholds = np.maximum(
        0.5 * np.std(train_delta, axis=(0, 2)),
        1e-6
    )

    def encode_events(X):

        encoded = []

        for trial in X:

            sampled = resample(
                trial,
                STEPS,
                axis=1
            )

            delta = np.diff(
                sampled,
                axis=1,
                prepend=sampled[:, :1]
            )

            on = (
                delta > thresholds[:, None]
            ).astype(np.float32)

            off = (
                delta < -thresholds[:, None]
            ).astype(np.float32)

            events = np.concatenate(
                [on, off],
                axis=0
            ).T

            encoded.append(events)

        return np.asarray(
            encoded,
            dtype=np.float32
        )

    return (
        encode_events(X_train_scaled),
        encode_events(X_test_scaled)
    )


# ============================================================
# 6. RUN-WISE VALIDATION
# ============================================================

def main():

    global y_train_fold

    X, y, run_ids = load_trials()

    logo = LeaveOneGroupOut()

    all_true = []
    lda_predictions = []
    snn_predictions = []

    fold_results = []

    unique_runs = np.unique(run_ids)

    print("\nStarting Leave-One-Run-Out validation")
    print("Number of folds:", len(unique_runs))

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=run_ids),
        start=1
    ):

        held_out_run = run_ids[test_idx][0]

        print("\n" + "=" * 55)
        print(f"FOLD {fold}/{len(unique_runs)}")
        print("Held-out recording run:", held_out_run)
        print("Training trials:", len(train_idx))
        print("Testing trials:", len(test_idx))
        print("=" * 55)

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        y_train_fold = y_train

        # ----------------------------------------------------
        # A. CSP + LDA BASELINE
        # ----------------------------------------------------

        print("\nTraining CSP + LDA baseline...")

        csp_lda = CSP(
            n_components=N_CSP_COMPONENTS,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False
        )

        train_features = csp_lda.fit_transform(
            X_train,
            y_train
        )

        test_features = csp_lda.transform(X_test)

        lda = LinearDiscriminantAnalysis()
        lda.fit(train_features, y_train)

        lda_pred = lda.predict(test_features)

        lda_accuracy = accuracy_score(
            y_test,
            lda_pred
        )

        lda_f1 = f1_score(
            y_test,
            lda_pred,
            average="macro",
            zero_division=0
        )

        print(
            f"CSP + LDA | Accuracy: {lda_accuracy:.4f}"
            f" | Macro-F1: {lda_f1:.4f}"
        )

        # ----------------------------------------------------
        # B. HYBRID CSP-SNN
        # ----------------------------------------------------

        print("\nPreparing hybrid CSP-SNN data...")

        X_train_events, X_test_events = prepare_snn_data(
            X_train,
            X_test
        )

        X_train_tensor = torch.tensor(
            X_train_events,
            dtype=torch.float32,
            device=device
        )

        X_test_tensor = torch.tensor(
            X_test_events,
            dtype=torch.float32,
            device=device
        )

        y_train_tensor = torch.tensor(
            y_train,
            dtype=torch.long,
            device=device
        )

        # Reset random seeds for reproducible fold initialization
        fold_seed = SEED + fold
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)

        if torch.backends.mps.is_available():
            torch.mps.manual_seed(fold_seed)

        model = HybridCSPSNN().to(device)

        print("\nTraining hybrid CSP-SNN...")

        start_time = time.perf_counter()

        train_snn(
            model,
            X_train_tensor,
            y_train_tensor
        )

        training_time = time.perf_counter() - start_time

        # Evaluate held-out run
        model.eval()

        with torch.no_grad():

            logits, spikes = model(
                X_test_tensor,
                return_activity=True
            )

            snn_pred = logits.argmax(dim=1).cpu().numpy()

            spike_density = float(
                spikes.mean().item()
            )

        snn_accuracy = accuracy_score(
            y_test,
            snn_pred
        )

        snn_f1 = f1_score(
            y_test,
            snn_pred,
            average="macro",
            zero_division=0
        )

        print(
            f"Hybrid CSP-SNN | Accuracy: {snn_accuracy:.4f}"
            f" | Macro-F1: {snn_f1:.4f}"
        )

        print(f"Hidden spike density: {spike_density:.5f}")
        print(f"Training time: {training_time:.2f} seconds")

        # ----------------------------------------------------
        # C. STORE FOLD RESULTS
        # ----------------------------------------------------

        all_true.extend(y_test)
        lda_predictions.extend(lda_pred)
        snn_predictions.extend(snn_pred)

        fold_results.append({
            "fold": fold,
            "held_out_run": int(held_out_run),
            "test_trials": len(test_idx),
            "lda_accuracy": lda_accuracy,
            "lda_macro_f1": lda_f1,
            "snn_accuracy": snn_accuracy,
            "snn_macro_f1": snn_f1,
            "snn_spike_density": spike_density,
            "snn_training_time_seconds": training_time
        })

    # ========================================================
    # 7. AGGREGATE RESULTS
    # ========================================================

    all_true = np.asarray(all_true)
    lda_predictions = np.asarray(lda_predictions)
    snn_predictions = np.asarray(snn_predictions)

    results = pd.DataFrame(fold_results)

    results.to_csv(
        f"{OUTPUT_DIR}/fold_results.csv",
        index=False
    )

    print("\n\n" + "=" * 65)
    print("RUN-WISE VALIDATION SUMMARY")
    print("=" * 65)

    print("\nFold-level results:")
    print(results.to_string(index=False))

    print("\nMean and standard deviation:")
    print(
        results[
            [
                "lda_accuracy",
                "lda_macro_f1",
                "snn_accuracy",
                "snn_macro_f1",
                "snn_spike_density"
            ]
        ].agg(["mean", "std"]).round(4)
    )

    # Pooled predictions: every trial is tested exactly once
    lda_pooled_accuracy = accuracy_score(
        all_true,
        lda_predictions
    )

    snn_pooled_accuracy = accuracy_score(
        all_true,
        snn_predictions
    )

    lda_pooled_f1 = f1_score(
        all_true,
        lda_predictions,
        average="macro",
        zero_division=0
    )

    snn_pooled_f1 = f1_score(
        all_true,
        snn_predictions,
        average="macro",
        zero_division=0
    )

    print("\nPooled out-of-run performance:")
    print(
        f"CSP + LDA | Accuracy: {lda_pooled_accuracy:.4f}"
        f" | Macro-F1: {lda_pooled_f1:.4f}"
    )

    print(
        f"Hybrid SNN | Accuracy: {snn_pooled_accuracy:.4f}"
        f" | Macro-F1: {snn_pooled_f1:.4f}"
    )

    # ========================================================
    # 8. CLASSIFICATION REPORTS
    # ========================================================

    lda_report = classification_report(
        all_true,
        lda_predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    snn_report = classification_report(
        all_true,
        snn_predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nCSP + LDA classification report:")
    print(lda_report)

    print("\nHybrid CSP-SNN classification report:")
    print(snn_report)

    # ========================================================
    # 9. SAVE POOLED PREDICTIONS AND REPORT
    # ========================================================

    pd.DataFrame({
        "true_label": all_true,
        "lda_prediction": lda_predictions,
        "snn_prediction": snn_predictions
    }).to_csv(
        f"{OUTPUT_DIR}/pooled_predictions.csv",
        index=False
    )

    with open(
        f"{OUTPUT_DIR}/validation_summary.txt",
        "w"
    ) as f:

        f.write("Leave-One-Run-Out Validation\n")
        f.write("============================\n\n")

        f.write(f"Total trials: {len(all_true)}\n")
        f.write(f"Number of held-out runs: {len(unique_runs)}\n\n")

        f.write("CSP + LDA\n")
        f.write(f"Pooled accuracy: {lda_pooled_accuracy:.4f}\n")
        f.write(f"Pooled macro-F1: {lda_pooled_f1:.4f}\n\n")
        f.write(lda_report)

        f.write("\n\nHybrid CSP-SNN\n")
        f.write(f"Pooled accuracy: {snn_pooled_accuracy:.4f}\n")
        f.write(f"Pooled macro-F1: {snn_pooled_f1:.4f}\n\n")
        f.write(snn_report)

        f.write("\n\nFold-level results\n")
        f.write(results.to_string(index=False))

    # ========================================================
    # 10. CONFUSION MATRICES
    # ========================================================

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, predictions, title in [
        (axes[0], lda_predictions, "CSP + LDA"),
        (axes[1], snn_predictions, "Hybrid CSP-SNN")
    ]:

        cm = confusion_matrix(
            all_true,
            predictions,
            labels=[0, 1, 2, 3]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASS_NAMES
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format="d"
        )

        ax.set_title(title)

    plt.tight_layout()

    plt.savefig(
        "results/figures/runwise_confusion_matrices.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # 11. FOLD ACCURACY COMPARISON
    # ========================================================

    x = np.arange(len(results))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - width / 2,
        results["lda_accuracy"],
        width,
        label="CSP + LDA"
    )

    plt.bar(
        x + width / 2,
        results["snn_accuracy"],
        width,
        label="Hybrid CSP-SNN"
    )

    plt.xticks(
        x,
        [f"Run {r}" for r in results["held_out_run"]]
    )

    plt.ylabel("Classification accuracy")
    plt.xlabel("Held-out recording run")
    plt.ylim(0, 1)
    plt.title("Run-Wise Model Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/runwise_accuracy_comparison.png",
        dpi=300
    )

    plt.close()

    print("\nResults saved in:")
    print(OUTPUT_DIR)
    print("Figures saved in results/figures/")
    print("\nRUN-WISE VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
