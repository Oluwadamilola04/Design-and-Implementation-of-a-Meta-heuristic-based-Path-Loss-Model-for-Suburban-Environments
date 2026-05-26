# Path Loss Model Optimization System

Radio propagation prediction for suburban environments using empirical models, metaheuristic optimization, and a Streamlit UI.

## Current Status

This project now runs successfully in the `pathloss` Conda environment using Python 3.11 and TensorFlow 2.15.1. The full analysis runner was executed on May 26, 2026 using all three available field datasets:

- `Yaba_updated_with_heights_and_freq.xlsx`
- `Sango Ota.xlsx`
- `Ijebu_ode_updated_with_heights_and_freq.xlsx`

The combined run used 3,616 measurement samples.

## Environment Setup

Use a dedicated Conda environment instead of Anaconda `base`. This avoids mixing packages from `C:\Users\HP\AppData\Roaming\Python\Python312\site-packages`.

```powershell
conda create -n pathloss python=3.11 -y
conda activate pathloss
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m ipykernel install --user --name pathloss --display-name "Path Loss Python 3.11"
```

In VS Code, select the notebook kernel named `Path Loss Python 3.11`.

## Run The Analysis

Full configured run:

```powershell
conda activate pathloss
python run_analysis.py
```

Quick smoke run:

```powershell
conda activate pathloss
python run_analysis.py --fast
```

The runner writes outputs to `results/`:

- `model_comparison_results.csv`
- `path_loss_predictions.csv`
- `optimized_parameters.csv`
- `optimization_history.csv`

## Streamlit App

Start the web UI with:

```powershell
conda activate pathloss
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The app supports:

- Single point path-loss prediction
- Batch prediction across a distance range
- Uploaded Excel measurement comparison
- CSV export of prediction results

Expected upload columns:

```text
Frequency_MHz
Base_Station_Height_m
Mobile_Station_Height_m
Distance (m)
Path Loss (dB)
```

## Latest Full-Run Results

Results from the combined three-dataset run:

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| PSO-Optimized | 7.8345 | 5.8198 | 0.2632 |
| GA-Optimized | 7.8610 | 5.8138 | 0.2582 |
| Egli Model | 19.1030 | 15.0013 | -3.3804 |
| COST-231 Urban | 26.7700 | 24.1296 | -7.6022 |
| COST-231 Suburban | 31.3225 | 28.9849 | -10.7768 |
| Okumura-Hata Suburban | 39.6785 | 37.8328 | -17.8984 |

## Observations

- The optimized COST-231 variants are far better than the unoptimized empirical models on the combined dataset.
- PSO produced the best RMSE in the latest full run: 7.8345 dB.
- GA was close to PSO, with slightly lower MAE but slightly higher RMSE.
- The baseline empirical models show negative R2 on the combined data, meaning they perform worse than predicting the global mean path loss for this combined field dataset.
- PSO convergence flattened after roughly 650-800 iterations, so 1000 iterations is reasonable for final reporting but more than needed for quick checks.
- The previous `requirements.txt` pinned TensorFlow 2.13, which is not suitable for the Python 3.12 setup that caused the DLL/runtime issue. The project is now documented around Python 3.11 and TensorFlow 2.15.1.

## Recommendations

- Use `Path Loss Python 3.11` for the notebook and the `pathloss` Conda environment for terminal runs.
- Keep `base` Anaconda clean; install project packages into `pathloss`.
- Treat `run_analysis.py --fast` as a smoke test before running the full optimization.
- Report full-run results from `results/model_comparison_results.csv`, not from a quick smoke run.
- Consider train/test or site-held-out validation before making final research claims. The current run optimizes and evaluates on the same combined dataset.
- Investigate site-specific performance for Yaba, Sango Ota, and Ijebu Ode separately; the combined R2 suggests each location may need separate calibration or extra features.
- Add terrain/elevation-derived features if the ANN or optimized formula is expected to generalize beyond these three routes.
- Keep the Streamlit app focused on empirical comparison unless optimized parameters from `results/optimized_parameters.csv` are explicitly loaded into the UI.

## Project Files

- `Path_Loss_Enhanced.ipynb` - notebook workflow
- `run_analysis.py` - command-line workflow
- `app.py` - Streamlit interface
- `models.py` - empirical and wrapper model implementations
- `optimization.py` - PSO and GA optimizers
- `analysis.py` - metrics and diagnostics
- `config.yaml` - datasets and run configuration
- `requirements.txt` - working environment dependencies
