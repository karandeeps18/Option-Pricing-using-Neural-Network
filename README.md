# Gaussian Weighted Loss Functions for Neural Network Option Pricing

This repository contains the implementation of the methods described in the paper
*Gaussian Weighted Loss Functions for Neural Network Option Pricing*. The paper introduces a liquidity-informed, Gaussian-weighted training
objective (G-WMSE) that concentrates a neural network's training emphasis on the economically
relevant regions of the option surface for short-dated and at-the-money contracts, while preserving
global smoothness through a continuous, differentiable weighting scheme. The objective is studied
first on synthetic Black–Scholes data and then on SPY call-option snapshots, using a compact
feedforward network so that performance differences arise from the objective rather than from model
size.

## Paper

[Gaussian Weighted Loss Functions for Neural Network Option Pricing](<SSRN_URL_TODO>) (SSRN).

## Key results (as reported in the paper)

- Approximately 66% reduction in at-the-money RMSE for the Gaussian-weighted objective relative to
  the standard MSE baseline on SPY call options.
- A compact feedforward network with approximately 4.5k parameters (two hidden layers of 64 units).
- Throughput of approximately 1.1 million contracts per second in batch mode on a standard CPU.

## Repository structure

```
.
├── Notebooks/
│   ├── pricing_on_synthetic_data.ipynb    # synthetic Black–Scholes experiment
│   ├── NN_option_pricing_using_SPY.ipynb  # SPY experiment (objectives + no-arbitrage diagnostics)
│   └── SPY_EDA.ipynb                       # SPY market-microstructure exploratory analysis
├── src/
│   ├── config.py                           # loads MARKETDATA_TOKEN from .env
│   ├── inference_benchmark.py              # latency / throughput benchmark helpers
│   ├── data/
│   │   ├── fetch.py                        # marketdata.app client + daily snapshot ingestion
│   │   ├── clean.py                        # raw JSON -> cleaned feature table
│   │   └── test_clean.py                   # unit tests for the cleaning step
│   └── models/                             # trained checkpoints + inference metadata
├── data_pipeline.py                        # end-to-end ingest + clean entry point
├── vix_data.py                             # downloads the VIX series used as the volatility proxy
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

The `data/` and `outputs/` directories are not tracked in this repository (see the SPY experiment
notes below).

## Setup

Developed and run with Python 3.13.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The notebooks run with an IPython kernel (for example via VS Code or JupyterLab).

## Synthetic Black–Scholes experiment

`Notebooks/pricing_on_synthetic_data.ipynb` contains the implementation of the synthetic
experiment and requires no external data. Run it top to bottom: it generates a Black–Scholes
dataset via Sobol sampling, trains the baseline (MSE) and Gaussian-weighted (G-WMSE) models, and
generates the experiment's figures and tables.

## SPY experiment

`Notebooks/NN_option_pricing_using_SPY.ipynb` contains the implementation of the SPY experiment,
including the moneyness-region metrics and the no-arbitrage diagnostics.

This experiment depends on daily SPY option-chain snapshots sourced from OPRA via marketdata.app.
**That data is not included** in this repository for licensing and size reasons. To run the
experiment end to end you need:

1. A marketdata.app API token in a `.env` file at the repository root:

   ```
   MARKETDATA_TOKEN=your_token_here
   ```

2. The ingestion and cleaning pipeline, which fetches and processes the snapshots:

   ```
   python data_pipeline.py
   python vix_data.py
   ```

Trained model checkpoints used for inference are provided under `src/models/`.

## Citation

If you use this code or the methods it implements, please cite the paper:

```bibtex
@article{sonewane2026gaussian,
  title  = {Gaussian Weighted Loss Functions for Neural Network Option Pricing},
  author = {Sonewane, Karandeep},
  year   = {2026},
  note   = {SSRN preprint},
  url    = {<SSRN_URL_TODO>}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
