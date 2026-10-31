# Closed-Loop Neuromorphic BCI for Adaptive Upper-Limb Prosthetic Control

**An exploratory hybrid brain–computer interface research prototype using motor-imagery EEG, event-driven encoding, and spiking neural networks.**

## Overview

This project investigates the potential of event-driven neuromorphic computing for motor-imagery brain–computer interfaces (BCIs), with a long-term application in adaptive upper-limb prosthetic control.

The prototype processes EEG signals from four motor-imagery classes—left hand, right hand, feet, and tongue—and evaluates a hybrid architecture combining Common Spatial Patterns (CSP) with a recurrent Leaky Integrate-and-Fire (LIF) spiking neural network (SNN).

The central motivation is to explore whether spike-based representations and sparse neural activity can offer a useful computational alternative to conventional machine-learning approaches for BCI applications.

**Project status:** Preliminary research prototype. The current implementation evaluates offline motor-imagery classification; it is not a clinically validated or physically deployed prosthetic-control system.

## Research objectives

- Develop an EEG preprocessing and motor-imagery classification pipeline.
- Convert continuous EEG features into event-based ON/OFF spike representations.
- Implement a recurrent LIF spiking neural network.
- Compare the hybrid SNN against a conventional CSP–LDA baseline.
- Evaluate performance using run-wise validation rather than relying only on random trial splits.
- Record computational indicators such as spike density and inference time as preliminary measures for future efficiency analysis.

## Dataset

**BCI Competition IV, Dataset 2a**

The project uses the training recording for participant A01 (`A01T.mat`).

Dataset characteristics:

- 22 EEG channels.
- 3 EOG channels (not used as model input).
- Sampling frequency: 250 Hz.
- Four motor-imagery classes: left hand, right hand, feet, and tongue.
- Six labeled recording runs used for the current evaluation.

After preprocessing and artifact-related trial exclusion, the working dataset contains **273 trials**, with a three-second motor-imagery segment per trial.

The raw dataset is not included in this repository. Obtain it through the official BCI Competition IV dataset distribution and place `A01T.mat` in:

```text
data/raw/A01T.mat
```

Please follow the dataset provider's terms of use and citation requirements.

## Methodology

### 1. EEG preprocessing

- Load the BCI Competition IV Dataset 2a recording.
- Select the 22 EEG channels.
- Apply the project’s band-pass preprocessing.
- Extract three-second motor-imagery trial segments.
- Exclude trials identified as unsuitable during preprocessing.

### 2. Conventional baseline: CSP + LDA

A Common Spatial Patterns (CSP) transformation extracts spatial features from EEG signals. Linear Discriminant Analysis (LDA) then performs four-class classification.

This provides a conventional machine-learning reference for evaluating the hybrid SNN.

### 3. Event-driven encoding

The hybrid pipeline projects EEG signals into CSP components and converts changes in the resulting signals into ON/OFF events using delta-based encoding.

The event representation is resampled into 100 time steps and provides 12 event channels to the SNN.

### 4. Spiking neural network

The SNN uses:

- Leaky Integrate-and-Fire (LIF) neuronal dynamics.
- A recurrent hidden layer with 64 units.
- Spike-rate and membrane-activity information for classification.
- Surrogate-gradient supervised training in the current classification pipeline.

An STDP weight-update demonstration is also included. However, the STDP update is **not yet integrated into the evaluated classification pipeline** and does not contribute to the reported SNN classification results.

## Evaluation

Two evaluation approaches were explored:

1. **Random stratified trial split:** an initial 75/25 train/test experiment.
2. **Leave-One-Run-Out validation:** six-fold evaluation, holding out one labeled recording run at a time and training on the other five runs.

The run-wise evaluation is the more informative result reported below because each test fold contains a recording run not used for training.

### Run-wise validation results

| Metric | CSP + LDA | Hybrid CSP–SNN |
|---|---:|---:|
| Pooled accuracy | 74.36% | 72.53% |
| Pooled macro-F1 | 0.7396 | 0.7211 |
| Mean fold accuracy | 74.41% | 72.54% |
| Fold accuracy standard deviation | 4.67 percentage points | 2.12 percentage points |

The hybrid SNN achieved an accuracy of **72.53%** and macro-F1 of **0.7211** across pooled out-of-run predictions. Its accuracy was 1.83 percentage points below the CSP–LDA baseline.

The SNN performed comparatively well on left- and right-hand imagery, while feet imagery remained a more challenging class. These results suggest that the hybrid approach is promising enough to investigate further, but do not establish superiority over the conventional baseline.

### Computational indicators

For the initial hybrid pipeline experiment, the recorded hidden-layer spike density was approximately **0.099**. The measured batch inference time was approximately **23.12 ms**, or **0.335 ms per trial when amortized across the batch**.

These are preliminary measurements from the implemented offline pipeline. They should not be interpreted as real-time streaming latency or proof of lower energy consumption. A fair energy comparison would require hardware-level power measurements and a carefully matched computational baseline.

## Repository structure

```text
neuromorphic-bci-prosthetics/
│
├── data/
│   └── raw/
│       └── A01T.mat              # Download separately; not tracked by Git
│
├── src/
│   ├── baseline.py               # CSP + LDA baseline
│   ├── pipeline.py               # Hybrid CSP–SNN pipeline
│   └── validate_runs.py           # Leave-One-Run-Out evaluation
│
├── results/
│   ├── figures/
│   │   ├── runwise_confusion_matrices.png
│   │   ├── runwise_accuracy_comparison.png
│   │   ├── hybrid_snn_confusion_matrix.png
│   │   └── hybrid_snn_training_loss.png
│   │
│   ├── metrics/
│   │   ├── hybrid_snn_predictions.csv
│   │   └── hybrid_snn_performance.txt
│   │
│   ├── models/
│   │   ├── hybrid_csp_snn.pth
│   │   ├── csp_filters.npy
│   │   ├── event_thresholds.npy
│   │   └── stdp_adapted_input_weights.pth
│   │
│   └── runwise_validation/
│       ├── fold_results.csv
│       ├── pooled_predictions.csv
│       └── validation_summary.txt
│
├── notebooks/
├── requirements.txt
├── .gitignore
└── README.md
```

The structure above describes the intended organization. Some generated files may be absent if a particular experiment has not been run.

## Installation and reproduction

The project was developed with Python 3.12.

Create and activate a Conda environment:

```bash
conda create -n neuromorphic-bci python=3.12 -y
conda activate neuromorphic-bci
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Place the downloaded dataset at `data/raw/A01T.mat`.

Run the experiments from the project root:

```bash
python src/baseline.py
python src/pipeline.py
python src/validate_runs.py
```

The scripts generate evaluation outputs and figures in the `results/` directory.

## Limitations

The current work has several important limitations:

- Evaluation is based on one participant's training recording, not a multi-participant study.
- Six held-out runs provide run-wise evaluation within one recording session, not subject-independent validation.
- The current SNN classifier is trained using surrogate-gradient backpropagation.
- STDP is demonstrated separately and has not yet been integrated into the evaluated classifier.
- The preprocessing pipeline includes offline zero-phase filtering, which is not strictly causal and must be replaced or adapted for streaming deployment.
- No physical prosthetic device, online EEG acquisition system, or clinical application has been implemented.
- Spike density and software inference time alone do not establish energy efficiency.

These limitations define the scope of the current prototype and guide the next research stages.

## Future work

The following are proposed next steps for anyone extending this project:

1. **Complete STDP integration:** implement a clearly defined STDP learning stage that actually updates the SNN weights used for classification.
2. **STDP ablation study:** compare otherwise equivalent models with and without STDP using the same run-wise folds and evaluation metrics.
3. **Online adaptation:** investigate whether the network can adapt to incoming EEG activity without using test labels, with a strict separation between training and evaluation.
4. **Causal signal processing:** replace offline zero-phase filtering with causal preprocessing suitable for real-time operation.
5. **Cross-subject validation:** evaluate generalization across multiple participants and sessions.
6. **Computational efficiency:** benchmark spike activity, synaptic operations, inference latency, and hardware-level energy consumption against appropriate conventional baselines.
7. **Prosthetic-control integration:** investigate real-time command generation, feedback, safety constraints, and eventual hardware integration.

## Research integrity note

The reported results reflect the experiments implemented in this repository. The project does not claim that STDP improves classification, that the SNN outperforms CSP–LDA, or that the current system achieves real-time or energy-efficient prosthetic control.

The purpose of this repository is to provide a reproducible starting point for further research into event-driven BCI systems and adaptive neuromorphic computing.

## Acknowledgements

This work uses the BCI Competition IV Dataset 2a. Please acknowledge and cite the original dataset providers when using the data in research or derivative work.

---

**Project:** Closed-Loop Event-Driven Neuromorphic BCI for Adaptive Upper-Limb Prosthetic Control via STDP  
**Implementation status:** Exploratory offline prototype  
**Primary focus:** Motor-imagery EEG, event-driven encoding, spiking neural networks, and future STDP-based adaptation