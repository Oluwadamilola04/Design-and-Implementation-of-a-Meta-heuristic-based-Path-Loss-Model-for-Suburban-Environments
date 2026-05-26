"""
Command-line runner for the path loss modeling workflow.

This mirrors the notebook workflow in a repeatable script:
- loads the configured Excel datasets
- evaluates empirical baseline models
- optimizes the COST-231 form with PSO and GA
- exports comparison metrics and predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from analysis import ComparisonReport, ModelEvaluator
from models import COST231HataSuburban, COST231HataUrban, EgliModel, OkumuraHataSuburban
from optimization import GAOptimizer, PSOptimizer, create_objective_function


REQUIRED_COLUMNS = [
    "Frequency_MHz",
    "Base_Station_Height_m",
    "Mobile_Station_Height_m",
    "Distance (m)",
    "Path Loss (dB)",
]


def cost231_optimized_formula(
    f_mhz: np.ndarray,
    h_b_m: np.ndarray,
    h_m_m: np.ndarray,
    d_km: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """COST-231 style model with nine optimizable coefficients."""
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = params

    d_km_safe = np.asarray(d_km) + 1e-9
    h_b_m_safe = np.asarray(h_b_m) + 1e-9
    f_mhz_arr = np.asarray(f_mhz)
    h_m_m_arr = np.asarray(h_m_m)

    a_hm_f = (c4 * np.log10(f_mhz_arr) - c5) * h_m_m_arr - (
        c6 * np.log10(f_mhz_arr) - c7
    )

    return (
        c1
        + c2 * np.log10(f_mhz_arr)
        - c3 * np.log10(h_b_m_safe)
        - a_hm_f
        + (c8 - c9 * np.log10(h_b_m_safe)) * np.log10(d_km_safe)
    )


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_measurements(config: dict, root: Path) -> pd.DataFrame:
    frames = []
    for dataset_name in config["data"]["datasets"]:
        dataset_path = root / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Configured dataset was not found: {dataset_path}")

        frame = pd.read_excel(dataset_path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{dataset_name} is missing required columns: {missing}")

        frame = frame.copy()
        frame["Dataset"] = dataset_path.stem
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run path loss model analysis")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced PSO/GA iterations for a quick smoke run",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)

    np.random.seed(config.get("random_state", 42))

    df = load_measurements(config, root)
    f_mhz = df["Frequency_MHz"].to_numpy(dtype=float)
    h_b_m = df["Base_Station_Height_m"].to_numpy(dtype=float)
    h_m_m = df["Mobile_Station_Height_m"].to_numpy(dtype=float)
    d_km = df["Distance (m)"].to_numpy(dtype=float) / 1000.0
    measured = df["Path Loss (dB)"].to_numpy(dtype=float)

    baseline_models = {
        "COST-231 Urban": COST231HataUrban(),
        "COST-231 Suburban": COST231HataSuburban(),
        "Okumura-Hata Suburban": OkumuraHataSuburban(),
        "Egli Model": EgliModel(),
    }

    predictions = {}
    comparison_input = {}
    for name, model in baseline_models.items():
        y_pred = model.predict(f_mhz, h_b_m, h_m_m, d_km)
        predictions[name] = y_pred
        comparison_input[name] = (y_pred, model)

    data_tuple = (f_mhz, h_b_m, h_m_m, d_km, measured)
    objective = create_objective_function(cost231_optimized_formula)

    pso_config = config["pso"]
    pso_iterations = 100 if args.fast else pso_config["n_iterations"]
    pso_particles = 20 if args.fast else pso_config["n_particles"]
    pso = PSOptimizer(
        n_particles=pso_particles,
        n_dimensions=pso_config["n_dimensions"],
        n_iterations=pso_iterations,
        c1=pso_config["cognitive_parameter"],
        c2=pso_config["social_parameter"],
        w=pso_config["inertia_weight"],
        bounds=(
            np.array(pso_config["bounds"]["lower"], dtype=float),
            np.array(pso_config["bounds"]["upper"], dtype=float),
        ),
    )
    pso_cost, pso_params, pso_history = pso.optimize(objective, data_tuple)
    pso_pred = cost231_optimized_formula(f_mhz, h_b_m, h_m_m, d_km, pso_params)
    predictions["PSO-Optimized"] = pso_pred
    comparison_input["PSO-Optimized"] = (pso_pred, pso)

    ga_config = config["ga"]
    ga_generations = 100 if args.fast else ga_config["n_generations"]
    ga_population = 40 if args.fast else ga_config["population_size"]
    ga = GAOptimizer(
        pop_size=ga_population,
        n_generations=ga_generations,
        cxpb=ga_config["crossover_probability"],
        mutpb=ga_config["mutation_probability"],
        bounds=(
            np.array(ga_config["bounds"]["lower"], dtype=float),
            np.array(ga_config["bounds"]["upper"], dtype=float),
        ),
    )
    ga_cost, ga_params, ga_history = ga.optimize(objective, data_tuple)
    ga_pred = cost231_optimized_formula(f_mhz, h_b_m, h_m_m, d_km, ga_params)
    predictions["GA-Optimized"] = ga_pred
    comparison_input["GA-Optimized"] = (ga_pred, ga)

    comparison_df = ComparisonReport.generate_comparison_table(comparison_input, measured)
    output_dir = root / config["output"]["results_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_features = np.column_stack([f_mhz, h_b_m, h_m_m, d_km])
    X_train, X_test, y_train, y_test = train_test_split(
        ann_features, measured, test_size=0.2, random_state=config.get("random_state", 42)
    )
    ann_scaler = StandardScaler()
    X_train_scaled = ann_scaler.fit_transform(X_train)
    X_test_scaled = ann_scaler.transform(X_test)
    X_all_scaled = ann_scaler.transform(ann_features)

    tf.keras.utils.set_random_seed(config.get("random_state", 42))
    ann_model = keras.Sequential(
        [
            keras.Input(shape=(X_train_scaled.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.10),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="linear"),
        ]
    )
    ann_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["ann"]["learning_rate"]),
        loss=config["ann"]["loss"],
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config["ann"]["early_stopping"]["patience"],
        restore_best_weights=True,
    )
    ann_model.fit(
        X_train_scaled,
        y_train,
        validation_split=config["ann"]["validation_split"],
        epochs=config["ann"]["epochs"] if not args.fast else 30,
        batch_size=config["ann"]["batch_size"],
        callbacks=[early_stopping],
        verbose=0,
    )
    ann_pred = ann_model.predict(X_all_scaled, verbose=0).flatten()
    ann_test_pred = ann_model.predict(X_test_scaled, verbose=0).flatten()
    ann_metrics = ModelEvaluator.calculate_metrics(measured, ann_pred)
    ann_test_metrics = ModelEvaluator.calculate_metrics(y_test, ann_test_pred)

    predictions["ANN"] = ann_pred
    comparison_input["ANN"] = (ann_pred, ann_model)
    comparison_df = ComparisonReport.generate_comparison_table(comparison_input, measured)

    comparison_path = output_dir / "model_comparison_results.csv"
    predictions_path = output_dir / "path_loss_predictions.csv"
    params_path = output_dir / "optimized_parameters.csv"
    history_path = output_dir / "optimization_history.csv"
    ann_model_path = output_dir / "ann_model.keras"
    ann_scaler_path = output_dir / "ann_scaler.pkl"
    ann_metadata_path = output_dir / "ann_metadata.csv"

    comparison_df.to_csv(comparison_path, index=False)

    export_df = df[["Dataset", "Distance (m)", "Path Loss (dB)"]].copy()
    for name, y_pred in predictions.items():
        export_df[f"Predicted_{name}"] = y_pred
    export_df.to_csv(predictions_path, index=False)

    pd.DataFrame(
        [
            {"Model": "PSO-Optimized", "RMSE": pso_cost, **{f"p{i + 1}": v for i, v in enumerate(pso_params)}},
            {"Model": "GA-Optimized", "RMSE": ga_cost, **{f"p{i + 1}": v for i, v in enumerate(ga_params)}},
        ]
    ).to_csv(params_path, index=False)

    pd.DataFrame(
        {
            "Iteration": np.arange(max(len(pso_history), len(ga_history))),
            "PSO_RMSE": pd.Series(pso_history),
            "GA_RMSE": pd.Series(ga_history),
        }
    ).to_csv(history_path, index=False)

    ann_model.save(ann_model_path)
    joblib.dump(ann_scaler, ann_scaler_path)
    pd.DataFrame(
        [
            {
                "Model": "ANN",
                "Train_Samples": len(X_train),
                "Test_Samples": len(X_test),
                "Full_RMSE": ann_metrics["RMSE"],
                "Full_MAE": ann_metrics["MAE"],
                "Full_R2": ann_metrics["R2"],
                "Heldout_RMSE": ann_test_metrics["RMSE"],
                "Heldout_MAE": ann_test_metrics["MAE"],
                "Heldout_R2": ann_test_metrics["R2"],
            }
        ]
    ).to_csv(ann_metadata_path, index=False)

    print("\nPath loss analysis complete")
    print(f"Samples: {len(df)} across {df['Dataset'].nunique()} datasets")
    print(comparison_df[["Model", "RMSE", "MAE", "R2"]].round(4).to_string(index=False))
    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
