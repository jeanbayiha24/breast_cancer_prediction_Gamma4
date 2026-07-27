import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Breast Cancer Prediction - Gamma4",
    page_icon="🩺",
    layout="wide"
)

MODELS_DIR = "models"

FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Default values approximating dataset means, used to prefill the form
DEFAULT_VALUES = {
    'radius_mean': 14.13, 'texture_mean': 19.29, 'perimeter_mean': 91.97,
    'area_mean': 654.89, 'smoothness_mean': 0.0964, 'compactness_mean': 0.1043,
    'concavity_mean': 0.0888, 'concave points_mean': 0.0489, 'symmetry_mean': 0.1812,
    'fractal_dimension_mean': 0.0628, 'radius_se': 0.4052, 'texture_se': 1.2169,
    'perimeter_se': 2.8661, 'area_se': 40.34, 'smoothness_se': 0.0070,
    'compactness_se': 0.0254, 'concavity_se': 0.0319, 'concave points_se': 0.0118,
    'symmetry_se': 0.0205, 'fractal_dimension_se': 0.0038, 'radius_worst': 16.27,
    'texture_worst': 25.68, 'perimeter_worst': 107.26, 'area_worst': 880.58,
    'smoothness_worst': 0.1324, 'compactness_worst': 0.2543, 'concavity_worst': 0.2722,
    'concave points_worst': 0.1146, 'symmetry_worst': 0.2901, 'fractal_dimension_worst': 0.0839
}


@st.cache_resource
def load_artifacts():
    svc_path = os.path.join(MODELS_DIR, "svc_breast_cancer_model.joblib")
    xgb_path = os.path.join(MODELS_DIR, "xgboost_best_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")

    artifacts = {"svc": None, "xgb": None, "scaler": None, "errors": []}

    if os.path.exists(svc_path):
        artifacts["svc"] = joblib.load(svc_path)
    else:
        artifacts["errors"].append(f"Fichier manquant : {svc_path}")

    if os.path.exists(xgb_path):
        artifacts["xgb"] = joblib.load(xgb_path)
    else:
        artifacts["errors"].append(f"Fichier manquant : {xgb_path}")

    if os.path.exists(scaler_path):
        artifacts["scaler"] = joblib.load(scaler_path)
    else:
        artifacts["errors"].append(f"Fichier manquant : {scaler_path} (nécessaire pour le SVC)")

    return artifacts


def predict(model_name, artifacts, input_df):
    """Retourne (label, proba_malignant) pour le modèle choisi."""
    if model_name == "SVC (Support Vector Machine)":
        model = artifacts["svc"]
        scaler = artifacts["scaler"]
        if model is None or scaler is None:
            return None, None
        X_scaled = scaler.transform(input_df)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]
    else:
        model = artifacts["xgb"]
        if model is None:
            return None, None
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

    label = "Malignant (M)" if pred == 1 else "Benign (B)"
    return label, proba


def main():
    st.title("🩺 Breast Cancer Diagnosis Prediction")
    st.markdown(
        "Cette application prédit si une masse mammaire est **maligne (M)** ou "
        "**bénigne (B)** à partir des caractéristiques cellulaires calculées sur "
        "une image FNA (Fine Needle Aspirate)."
    )

    artifacts = load_artifacts()

    if artifacts["errors"]:
        for err in artifacts["errors"]:
            st.error(err)
        st.info(
            "Assure-toi que les fichiers suivants existent dans le dossier `models/` : "
            "`svc_breast_cancer_model.joblib`, `xgboost_best_model.joblib`, `scaler.joblib`."
        )

    st.sidebar.header("⚙️ Configuration")
    model_choice = st.sidebar.selectbox(
        "Choisir le modèle de prédiction",
        ["SVC (Support Vector Machine)", "XGBoost"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**SVC** : Accuracy 0.974, Precision 1.00, Recall 0.929\n\n"
        "**XGBoost** : modèle ensemble sur features brutes (non standardisées)"
    )

    st.markdown("---")
    st.subheader("📋 Caractéristiques de la masse mammaire")
    st.caption("Ajuste les valeurs ci-dessous ou laisse les valeurs par défaut (moyennes du dataset).")

    input_values = {}

    tab_mean, tab_se, tab_worst = st.tabs(["Mean", "Standard Error (se)", "Worst"])

    mean_features = [f for f in FEATURES if f.endswith("_mean")]
    se_features = [f for f in FEATURES if f.endswith("_se")]
    worst_features = [f for f in FEATURES if f.endswith("_worst")]

    with tab_mean:
        cols = st.columns(3)
        for i, feat in enumerate(mean_features):
            with cols[i % 3]:
                input_values[feat] = st.number_input(
                    feat, value=float(DEFAULT_VALUES[feat]), format="%.5f", key=feat
                )

    with tab_se:
        cols = st.columns(3)
        for i, feat in enumerate(se_features):
            with cols[i % 3]:
                input_values[feat] = st.number_input(
                    feat, value=float(DEFAULT_VALUES[feat]), format="%.5f", key=feat
                )

    with tab_worst:
        cols = st.columns(3)
        for i, feat in enumerate(worst_features):
            with cols[i % 3]:
                input_values[feat] = st.number_input(
                    feat, value=float(DEFAULT_VALUES[feat]), format="%.5f", key=feat
                )

    st.markdown("---")

    if st.button("🔍 Predict Diagnosis", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_values])[FEATURES]

        label, proba = predict(model_choice, artifacts, input_df)

        if label is None:
            st.error("Impossible de faire la prédiction : modèle ou scaler manquant.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                if "Malignant" in label:
                    st.error(f"### Résultat : {label}")
                else:
                    st.success(f"### Résultat : {label}")

            with col2:
                st.metric(
                    label="Probabilité de malignité",
                    value=f"{proba * 100:.2f}%"
                )

            st.progress(min(max(proba, 0.0), 1.0))

            st.caption(
                f"Modèle utilisé : **{model_choice}**. "
                "Cette prédiction est fournie à titre indicatif et ne remplace pas un avis médical."
            )

    st.markdown("---")
    st.caption(
        "Dataset : Breast Cancer Wisconsin (Diagnostic). "
        "Modèles entraînés dans le cadre du capstone IIP Program."
    )


if __name__ == "__main__":
    main()
