
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, resample

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
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

for folder in [
    "results/models",
    "results/figures",
    "results/metrics",
]:
    os.makedirs(folder, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)


# ============================================================
# 2. LOAD EEG DATA
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
        positions = np.asarray(
            run["trial"]
        ).reshape(-1)

        artifacts = np.asarray(
            run["artifacts"]
        ).reshape(-1)

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

    print("Clean trials:", len(X))
    print("EEG shape:", X.shape)
    print("Class counts:", np.bincount(y))

    return X, y, run_ids


# ============================================================
# 3. CSP SPATIAL FILTERING
# ============================================================

def fit_csp_transform(
    X_train,
    y_train,
    X_test
):

    print("\nFitting CSP spatial filters...")

    csp = CSP(
        n_components=N_CSP_COMPONENTS,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False
    )

    # Fit only on training trials
    csp.fit(X_train, y_train)

    # Apply learned spatial filters to preserve time
    filters = csp.filters_[:N_CSP_COMPONENTS]

    X_train_projected = np.einsum(
        "kc,nct->nkt",
        filters,
        X_train
    )

    X_test_projected = np.einsum(
        "kc,nct->nkt",
        filters,
        X_test
    )

    print(
        "CSP projected training data:",
        X_train_projected.shape
    )

    print(
        "CSP projected test data:",
        X_test_projected.shape
    )

    return (
        X_train_projected,
        X_test_projected,
        csp
    )


# ============================================================
# 4. NORMALIZE CSP COMPONENTS
# ============================================================

def normalize_components(
    X_train,
    X_test
):

    print("\nNormalizing CSP components...")

    scaler = StandardScaler()

    # Fit using training trials and time points only
    train_flat = X_train.transpose(
        0, 2, 1
    ).reshape(-1, X_train.shape[1])

    scaler.fit(train_flat)

    def transform(X):

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

    return (
        transform(X_train),
        transform(X_test),
        scaler
    )


# ============================================================
# 5. TRAINING-DERIVED DELTA EVENT ENCODING
# ============================================================

def fit_event_thresholds(X_train):

    sampled = np.asarray([
        resample(
            trial,
            STEPS,
            axis=1
        )
        for trial in X_train
    ])

    delta = np.diff(
        sampled,
        axis=2,
        prepend=sampled[:, :, :1]
    )

    # One threshold per CSP component
    thresholds = (
        0.5 * np.std(delta, axis=(0, 2))
    )

    return np.maximum(
        thresholds,
        1e-6
    )


def encode_events(
    X,
    thresholds
):

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


# ============================================================
# 6. RECURRENT LIF SNN
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

    def forward(
        self,
        x,
        return_activity=False
    ):

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

            spk, mem = self.lif(
                current,
                mem
            )

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
            [
                spike_rates,
                membrane_activity
            ],
            dim=1
        )

        logits = self.classifier(features)

        if return_activity:
            return (
                logits,
                hidden_spikes,
                hidden_membranes
            )

        return logits


# ============================================================
# 7. TRAIN THE NETWORK
# ============================================================

def train_model(
    model,
    X_train,
    y_train
):

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

    losses = []

    for epoch in range(EPOCHS):

        model.train()

        permutation = torch.randperm(
            len(X_train),
            device=device
        )

        total_loss = 0.0

        for start in range(
            0,
            len(X_train),
            BATCH_SIZE
        ):

            indices = permutation[
                start:start + BATCH_SIZE
            ]

            batch_x = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += (
                loss.item() * len(indices)
            )

        average_loss = (
            total_loss / len(X_train)
        )

        losses.append(average_loss)

        scheduler.step(average_loss)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Loss: {average_loss:.4f}"
        )

    return losses


# ============================================================
# 8. STDP WEIGHT UPDATE
# ============================================================

@torch.no_grad()
def stdp_update(
    model,
    sample_x,
    learning_rate=0.001
):

    model.eval()

    _, spikes, _ = model(
        sample_x,
        return_activity=True
    )

    input_size = sample_x.shape[2]

    pre_trace = torch.zeros(
        sample_x.shape[0],
        input_size,
        device=device
    )

    post_trace = torch.zeros(
        sample_x.shape[0],
        HIDDEN_SIZE,
        device=device
    )

    weight_change = torch.zeros_like(
        model.input_layer.weight
    )

    for t in range(sample_x.shape[1]):

        pre = sample_x[:, t]
        post = spikes[:, t]

        weight_change += (
            post.T @ pre_trace
            - post_trace.T @ pre
        )

        pre_trace = 0.9 * pre_trace + pre
        post_trace = 0.9 * post_trace + post

    weight_change /= (
        sample_x.shape[0]
        * sample_x.shape[1]
    )

    adapted_weights = (
        model.input_layer.weight.detach()
        + learning_rate * weight_change
    )

    return adapted_weights, weight_change


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    model.eval()

    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        logits, spikes, membranes = model(
            X_test,
            return_activity=True
        )

        predictions = logits.argmax(dim=1)

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    y_pred = predictions.cpu().numpy()
    y_true = np.asarray(y_test)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    spike_density = float(
        spikes.mean().item()
    )

    print("\nHYBRID CSP-SNN TEST RESULTS")
    print("----------------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Hidden spike density: {spike_density:.5f}")
    print(
        "Batch inference time (ms):",
        round(elapsed * 1000, 3)
    )
    print(
        "Amortized time per trial (ms):",
        round(elapsed * 1000 / len(X_test), 4)
    )

    print("\nClassification report:")
    print(report)

    return (
        y_true,
        y_pred,
        accuracy,
        report,
        elapsed,
        spike_density
    )


# ============================================================
# 10. SAVE RESULTS
# ============================================================

def save_results(
    model,
    csp,
    scaler,
    thresholds,
    losses,
    y_true,
    y_pred,
    accuracy,
    report,
    elapsed,
    spike_density,
    training_time,
    stdp_change
):

    torch.save(
        model.state_dict(),
        "results/models/hybrid_csp_snn.pth"
    )

    np.save(
        "results/models/csp_filters.npy",
        csp.filters_
    )

    np.save(
        "results/models/event_thresholds.npy",
        thresholds
    )

    pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred
    }).to_csv(
        "results/metrics/hybrid_snn_predictions.csv",
        index=False
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Hybrid CSP-SNN Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "results/figures/hybrid_snn_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(losses) + 1),
        losses,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Hybrid CSP-SNN Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "results/figures/hybrid_snn_training_loss.png",
        dpi=300
    )

    plt.close()

    with open(
        "results/metrics/hybrid_snn_performance.txt",
        "w"
    ) as f:

        f.write("Hybrid CSP-Recurrent LIF SNN\n")
        f.write("============================\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Training time (s): {training_time:.3f}\n")
        f.write(f"Batch inference time (ms): {elapsed * 1000:.4f}\n")
        f.write(f"Hidden spike density: {spike_density:.5f}\n")
        f.write(f"STDP mean absolute update: {stdp_change:.8f}\n\n")
        f.write(report)

    print("\nResults saved.")


# ============================================================
# 11. MAIN
# ============================================================

def main():

    # Load data
    X, y, run_ids = load_trials()

    # Reproduce the same trial-level split as baseline
    indices = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=SEED,
        stratify=y
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # CSP fitting and projection
    (
        X_train_csp,
        X_test_csp,
        csp
    ) = fit_csp_transform(
        X_train,
        y_train,
        X_test
    )

    # Normalize CSP component signals
    (
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = normalize_components(
        X_train_csp,
        X_test_csp
    )

    # Learn event thresholds on training data only
    thresholds = fit_event_thresholds(
        X_train_scaled
    )

    print("\nCSP event thresholds:", thresholds)

    # Encode event sequences
    X_train_events = encode_events(
        X_train_scaled,
        thresholds
    )

    X_test_events = encode_events(
        X_test_scaled,
        thresholds
    )

    print(
        "Training event data:",
        X_train_events.shape
    )

    print(
        "Test event data:",
        X_test_events.shape
    )

    print(
        "Input event density:",
        round(float(X_train_events.mean()), 5)
    )

    # Convert to tensors
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

    # Initialize model
    model = HybridCSPSNN().to(device)

    print("\nTraining hybrid CSP-SNN...")

    training_start = time.perf_counter()

    losses = train_model(
        model,
        X_train_tensor,
        y_train_tensor
    )

    training_time = (
        time.perf_counter() - training_start
    )

    # Evaluate
    (
        y_true,
        y_pred,
        accuracy,
        report,
        elapsed,
        spike_density
    ) = evaluate_model(
        model,
        X_test_tensor,
        y_test
    )

    # STDP demonstration
    print("\nRunning STDP demonstration...")

    adapted_weights, weight_change = stdp_update(
        model,
        X_train_tensor[:16]
    )

    torch.save(
        adapted_weights.cpu(),
        "results/models/stdp_adapted_input_weights.pth"
    )

    stdp_change = float(
        weight_change.abs().mean().item()
    )

    print(
        "STDP mean absolute weight change:",
        f"{stdp_change:.8f}"
    )

    # Export everything
    save_results(
        model,
        csp,
        scaler,
        thresholds,
        losses,
        y_true,
        y_pred,
        accuracy,
        report,
        elapsed,
        spike_density,
        training_time,
        stdp_change
    )

    print("\n================================")
    print("HYBRID CSP-SNN PIPELINE COMPLETE")
    print("================================")


if __name__ == "__main__":
    main()
