# Hybrid Federated Learning Framework for Privacy-Preserving Smart Healthcare

MSc project implementing and evaluating a hybrid federated learning (FL) framework combining
layer-adaptive differential privacy (LAN-RDP), homomorphic encryption (CKKS), and lightweight
authenticated transport (ASCON-128), evaluated on the UCI MHEALTH wearable-sensor dataset.

## What this repository contains

- **`notebooks/mhealth_FL_framework.ipynb`** — the complete, self-contained implementation and
  experimental log. Run top to bottom in Google Colab or a local Jupyter environment (see
  Setup below). Covers, in order:
  1. Data preprocessing and windowing (MHEALTH dataset)
  2. Centralised LSTM baseline
  3. Federated baseline (FedAvg, no privacy mechanism)
  4. Per-layer Rényi Differential Privacy (RDP) accountant and budget calibration
  5–8. Per-example DP-SGD: toy-scale validation, scaled-up comparison, a stability-fix attempt,
     and the Adam-vs-SGD optimiser investigation
  9–10. CKKS homomorphic encryption + ASCON-128 transport layer, and its combination with
     top-k sparsification for communication-cost reduction
  11–12. Validating sparsification's accuracy cost, and testing whether it rescues LAN-RDP's
     stability
  13. Poisoning / Byzantine-robustness evaluation (label-flipping attack vs. Trimmed Mean)
  14. Weighted clip-norm split — the project's key revised finding for LAN-RDP

- **`docs/Design_Development_Evaluation.docx`** — the dissertation sections (Design, Development,
  Evaluation) describing the framework's architecture and a full write-up of every experiment
  and finding in the notebook, with critical reflection on limitations.

- **`data/`** — place the MHEALTH dataset here (see Data below); not included in this repository.

## Setup

### Option A: Google Colab (recommended, no local setup required)
1. Upload `notebooks/mhealth_FL_framework.ipynb` to Colab.
2. Upload the MHEALTH dataset zip when prompted in the first code cell (see Data below).
3. Run cells top to bottom. Package installs (`opacus`, `dp-accounting`, `tenseal`, `ascon`) are
   handled by `!pip install` cells within the notebook itself.

### Option B: Local Jupyter environment
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/mhealth_FL_framework.ipynb
```

**Note on PyTorch/CUDA:** this project runs entirely on CPU (`device = torch.device("cpu")`
throughout). If `pip install torch` on your platform pulls in CUDA-linked binaries that fail to
import, install the CPU-only build instead:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Data

This project uses the **MHEALTH dataset** (Banos et al., 2014), available from the UCI Machine
Learning Repository:
- Dataset page: https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- Citation: Banos, O., Garcia, R. & Saez, A. (2014). MHEALTH [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5TW22

Download the dataset zip and either:
- Upload it directly in the notebook's first code cell (Colab), or
- Extract it to `data/MHEALTHDATASET/` (local), matching the path the notebook expects.

The dataset is not included in this repository due to its size (~75 MB) and to respect the
original UCI licensing terms (CC BY 4.0 — redistribution is permitted, but linking to the
canonical source is preferred practice).


## Requirements

See `requirements.txt`. Core dependencies: PyTorch, Opacus (per-example DP-SGD), Google's
`dp-accounting` (RDP privacy accountant), TenSEAL (CKKS homomorphic encryption), `ascon`
(ASCON-128 reference implementation), scikit-learn, pandas, numpy.

## License

Code in this repository is provided for academic assessment purposes. The MHEALTH dataset is
licensed CC BY 4.0 by its original authors (Banos et al., 2014) and is not redistributed here.


