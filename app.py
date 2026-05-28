"""
Streamlit application for radio path-loss prediction and model comparison.
"""

import os
import sys
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras # type: ignore

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from models import (  # noqa: E402
    ANNModel,
    COST231HataSuburban,
    COST231HataUrban,
    EgliModel,
    OkumuraHataSuburban,
    OptimizedCOST231,
)


REQUIRED_COLUMNS = [
    "Frequency_MHz",
    "Base_Station_Height_m",
    "Mobile_Station_Height_m",
    "Distance (m)",
    "Path Loss (dB)",
]


st.set_page_config(
    page_title="Path Loss Prediction Tool",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("# 📡 Radio Path Loss Prediction Tool")
st.markdown(
    "Predict and compare radio path loss for suburban environments using "
    "empirical, optimized, and neural-network models."
)

st.sidebar.header("Configuration")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["Single Prediction", "Batch Analysis", "Model Comparison"],
)


@st.cache_resource
def load_models():
    """Load empirical models and saved optimized PSO/GA models when available."""
    loaded_models = {
        "COST-231 Urban": COST231HataUrban(),
        "COST-231 Suburban": COST231HataSuburban(),
        "Okumura-Hata Suburban": OkumuraHataSuburban(),
        "Egli Model": EgliModel(),
    }
    load_warnings = []

    params_path = os.path.join(project_dir, "results", "optimized_parameters.csv")
    if os.path.exists(params_path):
        try:
            params_df = pd.read_csv(params_path)
            param_cols = [f"p{i}" for i in range(1, 10)]
            for _, row in params_df.iterrows():
                if all(col in row.index for col in param_cols):
                    loaded_models[row["Model"]] = OptimizedCOST231(
                        row[param_cols].to_numpy(dtype=float)
                    )
        except Exception as exc:
            load_warnings.append(
                f"Saved optimized models could not be loaded from results/optimized_parameters.csv: {exc}"
            )

    ann_model_path = os.path.join(project_dir, "results", "ann_model.keras")
    ann_scaler_path = os.path.join(project_dir, "results", "ann_scaler.pkl")
    if os.path.exists(ann_model_path) and os.path.exists(ann_scaler_path):
        try:
            ann_model = keras.models.load_model(ann_model_path)
            ann_wrapper = ANNModel(ann_model)
            ann_wrapper.set_scaler(joblib.load(ann_scaler_path))
            loaded_models["ANN"] = ann_wrapper
        except Exception as exc:
            load_warnings.append(
                "Saved ANN artifacts could not be loaded on this deployment. "
                f"The app will continue without ANN in single/batch prediction."
            )

    return loaded_models, load_warnings


models, model_load_warnings = load_models()

for warning_message in model_load_warnings:
    st.warning(warning_message)


@st.cache_data
def load_training_domain():
    """Load feature ranges used by the saved ANN artifacts."""
    config_path = os.path.join(project_dir, "config.yaml")
    if not os.path.exists(config_path):
        return None

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    frames = []
    for dataset_name in config.get("data", {}).get("datasets", []):
        dataset_path = os.path.join(project_dir, dataset_name)
        if os.path.exists(dataset_path):
            frames.append(pd.read_excel(dataset_path, usecols=REQUIRED_COLUMNS))

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    return {
        "frequency": (df["Frequency_MHz"].min(), df["Frequency_MHz"].max()),
        "base_height": (df["Base_Station_Height_m"].min(), df["Base_Station_Height_m"].max()),
        "mobile_height": (df["Mobile_Station_Height_m"].min(), df["Mobile_Station_Height_m"].max()),
        "distance_km": (df["Distance (m)"].min() / 1000.0, df["Distance (m)"].max() / 1000.0),
    }


ann_domain = load_training_domain()


def calculate_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


@st.cache_resource(show_spinner=False)
def train_ann_model(dataframe_json):
    """Train an ANN on uploaded measurement data."""
    df = pd.read_json(StringIO(dataframe_json))
    features = np.column_stack(
        [
            df["Frequency_MHz"].to_numpy(dtype=float),
            df["Base_Station_Height_m"].to_numpy(dtype=float),
            df["Mobile_Station_Height_m"].to_numpy(dtype=float),
            df["Distance (m)"].to_numpy(dtype=float) / 1000.0,
        ]
    )
    target = df["Path Loss (dB)"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(features)

    tf.keras.utils.set_random_seed(42)
    model = keras.Sequential(
        [
            keras.Input(shape=(X_train_scaled.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.10),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
    )

    all_predictions = model.predict(X_all_scaled, verbose=0).flatten()
    test_predictions = model.predict(X_test_scaled, verbose=0).flatten()
    return {
        "predictions": all_predictions,
        "test_metrics": calculate_metrics(y_test, test_predictions),
        "epochs": len(history.history["loss"]),
    }


def predict_with_model(model, frequency, bs_height, mobile_height, distance_km):
    return model.predict(
        np.asarray(frequency),
        np.asarray(bs_height),
        np.asarray(mobile_height),
        np.asarray(distance_km),
    )


def ann_in_domain(frequency, bs_height, mobile_height, distance_km):
    """Return a boolean mask indicating which points are inside the ANN training range."""
    if ann_domain is None:
        return np.ones_like(np.asarray(distance_km, dtype=float), dtype=bool)

    frequency = np.asarray(frequency, dtype=float)
    bs_height = np.asarray(bs_height, dtype=float)
    mobile_height = np.asarray(mobile_height, dtype=float)
    distance_km = np.asarray(distance_km, dtype=float)

    return (
        (frequency >= ann_domain["frequency"][0])
        & (frequency <= ann_domain["frequency"][1])
        & (bs_height >= ann_domain["base_height"][0])
        & (bs_height <= ann_domain["base_height"][1])
        & (mobile_height >= ann_domain["mobile_height"][0])
        & (mobile_height <= ann_domain["mobile_height"][1])
        & (distance_km >= ann_domain["distance_km"][0])
        & (distance_km <= ann_domain["distance_km"][1])
    )


def ann_domain_message():
    if ann_domain is None:
        return "ANN training range is unavailable."

    return (
        "Saved ANN training range: "
        f"frequency {ann_domain['frequency'][0]:.0f}-{ann_domain['frequency'][1]:.0f} MHz, "
        f"base height {ann_domain['base_height'][0]:.1f}-{ann_domain['base_height'][1]:.1f} m, "
        f"mobile height {ann_domain['mobile_height'][0]:.1f}-{ann_domain['mobile_height'][1]:.1f} m, "
        f"distance {ann_domain['distance_km'][0]:.3f}-{ann_domain['distance_km'][1]:.3f} km."
    )


def plot_prediction_comparison(distances, predictions_dict, title="Path Loss vs Distance"):
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, path_losses in predictions_dict.items():
        ax.plot(distances, path_losses, marker="o", label=model_name, linewidth=2, markersize=5)

    ax.set_xlabel("Distance (km)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Path Loss (dB)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return fig


def plot_model_error_distribution(predictions_dict, measured_values, title="Error Distribution"):
    fig, ax = plt.subplots(figsize=(12, 6))
    errors = {model: measured_values - preds for model, preds in predictions_dict.items()}
    models_list = list(errors.keys())
    error_data = [errors[m] for m in models_list]

    bp = ax.boxplot(error_data, labels=models_list, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(error_data), 1)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Error (dB)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=20, ha="right")
    return fig


if app_mode == "Single Prediction":
    st.header("Single Point Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")
        frequency = st.number_input("Frequency (MHz)", value=1800.0, min_value=150.0, max_value=6000.0, step=10.0)
        bs_height = st.number_input("Base Station Height (m)", value=30.0, min_value=5.0, max_value=500.0, step=1.0)
        mobile_height = st.number_input("Mobile Station Height (m)", value=1.5, min_value=0.5, max_value=10.0, step=0.1)
        distance = st.number_input("Distance (km)", value=1.0, min_value=0.01, max_value=100.0, step=0.1)

    with col2:
        st.subheader("Selected Models")
        selected_models = st.multiselect(
            "Choose models to compare:",
            list(models.keys()),
            default=list(models.keys()),
            key="model_selection_single",
        )

        if selected_models:
            predictions = {}
            display_rows = []
            for model_name in selected_models:
                if model_name == "ANN" and not ann_in_domain([frequency], [bs_height], [mobile_height], [distance])[0]:
                    display_rows.append(
                        {
                            "Model": model_name,
                            "Path Loss (dB)": "Outside ANN training range",
                        }
                    )
                    st.warning(ann_domain_message())
                    continue

                prediction = predict_with_model(
                    models[model_name], [frequency], [bs_height], [mobile_height], [distance]
                )[0]
                predictions[model_name] = prediction
                display_rows.append({"Model": model_name, "Path Loss (dB)": f"{prediction:.2f}"})

            results_df = pd.DataFrame(display_rows)
            st.dataframe(results_df, width="stretch")

            pred_values = list(predictions.values())
            if pred_values:
                st.metric("Average Path Loss", f"{np.mean(pred_values):.2f} dB")

                c1, c2, c3 = st.columns(3)
                c1.metric("Min", f"{np.min(pred_values):.2f} dB")
                c2.metric("Max", f"{np.max(pred_values):.2f} dB")
                c3.metric("Std Dev", f"{np.std(pred_values):.2f} dB")


elif app_mode == "Batch Analysis":
    st.header("Batch Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        frequency = st.number_input("Frequency (MHz)", value=1800.0, min_value=150.0, max_value=6000.0, step=10.0)
        bs_height = st.number_input("Base Station Height (m)", value=30.0, min_value=5.0, max_value=500.0, step=1.0)
        mobile_height = st.number_input("Mobile Height (m)", value=1.5, min_value=0.5, max_value=10.0, step=0.1)

        st.subheader("Distance Range")
        range_col1, range_col2 = st.columns(2)
        distance_min = range_col1.number_input("Min (km)", value=0.1, min_value=0.01, step=0.1)
        distance_max = range_col2.number_input("Max (km)", value=10.0, min_value=0.1, step=0.1)
        n_points = st.slider("Number of Points", 5, 100, 20)

        selected_models_batch = st.multiselect(
            "Select Models:",
            list(models.keys()),
            default=list(models.keys()),
            key="model_selection_batch",
        )

    with col2:
        if selected_models_batch:
            distances = np.linspace(distance_min, distance_max, n_points)
            predictions_dict = {}
            for model_name in selected_models_batch:
                frequency_arr = np.full(n_points, frequency)
                bs_height_arr = np.full(n_points, bs_height)
                mobile_height_arr = np.full(n_points, mobile_height)

                if model_name == "ANN":
                    valid_mask = ann_in_domain(frequency_arr, bs_height_arr, mobile_height_arr, distances)
                    ann_predictions = np.full(n_points, np.nan)
                    if valid_mask.any():
                        ann_predictions[valid_mask] = predict_with_model(
                            models[model_name],
                            frequency_arr[valid_mask],
                            bs_height_arr[valid_mask],
                            mobile_height_arr[valid_mask],
                            distances[valid_mask],
                        )
                    if (~valid_mask).any():
                        st.warning(
                            "ANN predictions outside the saved training range are hidden. "
                            + ann_domain_message()
                        )
                    predictions_dict[model_name] = ann_predictions
                else:
                    predictions_dict[model_name] = predict_with_model(
                        models[model_name],
                        frequency_arr,
                        bs_height_arr,
                        mobile_height_arr,
                        distances,
                    )

            st.pyplot(plot_prediction_comparison(distances, predictions_dict), width="stretch")

            results_table = pd.DataFrame({"Distance (km)": distances})
            for model_name, path_losses in predictions_dict.items():
                results_table[f"{model_name} (dB)"] = path_losses.round(2)
            st.dataframe(results_table, width="stretch")

            st.download_button(
                label="Download Results (CSV)",
                data=results_table.to_csv(index=False),
                file_name="path_loss_predictions.csv",
                mime="text/csv",
            )


elif app_mode == "Model Comparison":
    st.header("Model Comparison & Analysis")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    with col2:
        include_ann = st.checkbox("Train ANN on uploaded data", value=True)
        st.info("Expected columns: " + ", ".join(REQUIRED_COLUMNS))

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()

            st.success("File loaded successfully")
            st.dataframe(df.head(), width="stretch")

            progress_bar = st.progress(0)
            status_text = st.empty()

            predictions_dict = {}
            metrics_dict = {}
            distances_km = df["Distance (m)"].to_numpy(dtype=float) / 1000.0
            measured = df["Path Loss (dB)"].to_numpy(dtype=float)
            comparison_models = list(models.items())

            for idx, (model_name, model) in enumerate(comparison_models):
                status_text.text(f"Processing: {model_name}")
                path_losses = predict_with_model(
                    model,
                    df["Frequency_MHz"].to_numpy(dtype=float),
                    df["Base_Station_Height_m"].to_numpy(dtype=float),
                    df["Mobile_Station_Height_m"].to_numpy(dtype=float),
                    distances_km,
                )
                predictions_dict[model_name] = path_losses
                metrics_dict[model_name] = calculate_metrics(measured, path_losses)
                progress_bar.progress((idx + 1) / (len(comparison_models) + int(include_ann)))

            if include_ann:
                status_text.text("Training ANN")
                ann_artifacts = train_ann_model(df[REQUIRED_COLUMNS].to_json())
                predictions_dict["ANN"] = ann_artifacts["predictions"]
                metrics_dict["ANN"] = {
                    **calculate_metrics(measured, ann_artifacts["predictions"]),
                    "Heldout_RMSE": ann_artifacts["test_metrics"]["RMSE"],
                    "Epochs": ann_artifacts["epochs"],
                }
                progress_bar.progress(1.0)

            status_text.text("Predictions complete")

            metrics_df = pd.DataFrame(metrics_dict).T.sort_values("RMSE")
            st.subheader("Performance Metrics")
            st.dataframe(metrics_df.round(4), width="stretch")

            chart_col1, chart_col2 = st.columns([1, 1])
            with chart_col1:
                st.subheader("Path Loss Comparison")
                st.pyplot(
                    plot_prediction_comparison(
                        distances_km,
                        {**predictions_dict, "Measured": measured},
                        "Predicted vs Measured Path Loss",
                    ),
                    width="stretch",
                )

            with chart_col2:
                st.subheader("Error Distribution")
                st.pyplot(
                    plot_model_error_distribution(
                        predictions_dict,
                        measured,
                        "Prediction Error Distribution",
                    ),
                    width="stretch",
                )

            st.subheader("Residuals Analysis")
            for model_name, predictions in predictions_dict.items():
                residuals = measured - predictions
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"{model_name} Mean Error", f"{np.mean(residuals):.2f} dB")
                m2.metric(f"{model_name} Std Dev", f"{np.std(residuals):.2f} dB")
                m3.metric(f"{model_name} Min Abs Error", f"{np.min(np.abs(residuals)):.2f} dB")
                m4.metric(f"{model_name} Max Abs Error", f"{np.max(np.abs(residuals)):.2f} dB")

            results_export = df.copy()
            for model_name, predictions in predictions_dict.items():
                results_export[f"Predicted_{model_name}"] = predictions.round(2)

            st.download_button(
                label="Download Analysis Results",
                data=results_export.to_csv(index=False),
                file_name="path_loss_analysis.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Error processing file: {exc}")


st.markdown("---")
st.markdown(
    """
### Model Information
- COST-231 Urban/Suburban, Okumura-Hata, and Egli are empirical baselines.
- PSO-Optimized and GA-Optimized are loaded from `results/optimized_parameters.csv` when available.
- ANN is trained inside Model Comparison using the uploaded measured dataset.
"""
)
