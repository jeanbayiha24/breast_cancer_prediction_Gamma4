import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

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

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 1rem;}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.85), rgba(255,244,240,0.95));
    border: 1px solid rgba(255, 133, 102, 0.18);
    padding: 1rem;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}
div.stButton > button {
    border-radius: 14px;
    height: 3rem;
    font-weight: 600;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff7f2 0%, #fffdfb 100%);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    svc_path = os.path.join(MODELS_DIR, "svc_breast_cancer_model.joblib")
    xgb_path = os.path.join(MODELS_DIR, "xgboost_best_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    xgb_scaler_path = os.path.join(MODELS_DIR, "xgb_scaler.joblib")
    xgb_features_path = os.path.join(MODELS_DIR, "xgb_selected_features.joblib")

    artifacts = {"svc": None, "xgb": None, "scaler": None, "xgb_scaler": None, "xgb_features": None, "errors": []}

    for key, path, msg in [
        ("svc", svc_path, svc_path),
        ("xgb", xgb_path, xgb_path),
        ("scaler", scaler_path, f"{scaler_path} (nécessaire pour le SVC)"),
        ("xgb_scaler", xgb_scaler_path, f"{xgb_scaler_path} (nécessaire pour XGBoost)"),
        ("xgb_features", xgb_features_path, xgb_features_path),
    ]:
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
        else:
            artifacts["errors"].append(f"Fichier manquant : {msg}")
    return artifacts

@st.cache_data
def load_dataset():
    data = load_breast_cancer()
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    df = pd.DataFrame(data.data, columns=feature_names)
    df["target"] = data.target
    df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
    return df


def predict(model_name, artifacts, input_df):
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
        xgb_scaler = artifacts["xgb_scaler"]
        xgb_features = artifacts["xgb_features"]
        if model is None or xgb_scaler is None or xgb_features is None:
            return None, None
        model_input = input_df[xgb_features]
        model_input_scaled = xgb_scaler.transform(model_input)
        pred = model.predict(model_input_scaled)[0]
        proba = model.predict_proba(model_input_scaled)[0][1]
    label = "Malignant (M)" if pred == 1 else "Benign (B)"
    return label, proba


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="diagnosis", palette=["#ef6c57", "#66bb8a"], ax=ax)
    ax.set_title("Diagnosis distribution", fontsize=13, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()), ha='center', va='bottom')
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df, full_matrix=False):
    cols = FEATURES + ["target"] if full_matrix else [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "radius_worst", "area_worst", "target"
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(14, 10) if full_matrix else (9, 6))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=not full_matrix,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.9},
        ax=ax
    )
    ax.set_title("Correlation Heatmap (numerical features)", fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    return fig


def plot_feature_distribution(df, scaled=False):
    selected = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "concavity_mean", "concave points_mean"]
    plot_df = df[[*selected, "diagnosis"]].copy()
    title = "Feature distributions by diagnosis"
    ylabel = "Value"
    if scaled:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        plot_df[selected] = scaler.fit_transform(plot_df[selected])
        title = "Feature distributions by diagnosis (scaled 0-1)"
        ylabel = "Scaled value"
    long_df = plot_df.melt(id_vars="diagnosis", value_vars=selected, var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=long_df,
        x="feature",
        y="value",
        hue="diagnosis",
        palette={"Malignant": "#d97767", "Benign": "#6fb08f"},
        width=0.68,
        fliersize=3,
        linewidth=1.2,
        ax=ax
    )
    ax.set_title(title, fontsize=22, fontweight='bold', pad=16)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis='x', rotation=20, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(title="diagnosis", fontsize=11, title_fontsize=12, frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_mean_comparison(df):
    compare = df.groupby("diagnosis")[["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]].mean().T
    fig, ax = plt.subplots(figsize=(7, 4.5))
    compare.plot(kind="bar", ax=ax, color=["#ef6c57", "#66bb8a"])
    ax.set_title("Mean feature comparison", fontsize=13, fontweight='bold')
    ax.set_ylabel("Average value")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    return fig

def plot_xgb_feature_importance(artifacts, top_n=15):
    """Retourne une figure Matplotlib montrant l'importance des variables XGBoost."""
    model = artifacts.get("xgb")
    xgb_features = artifacts.get("xgb_features")

    # Vérifications
    if model is None or xgb_features is None:
        return None

    import numpy as np
    import matplotlib.pyplot as plt

    # Importance des features (API scikit-learn de XGBoost)
    try:
        importances = model.feature_importances_
    except AttributeError:
        return None

    feature_names = list(xgb_features)
    idx = np.argsort(importances)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in idx]
    sorted_importances = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sorted_features[::-1], sorted_importances[::-1], color="#d97767")
    ax.set_title("XGBoost Feature Importance", fontsize=16, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    fig.tight_layout()
    return fig


def main():
    st.title("🩺 Breast Cancer Diagnosis Prediction")
    st.markdown("A cleaner Streamlit interface combining **prediction** and **interactive dataset insights**.")

    artifacts = load_artifacts()
    df = load_dataset()

    if artifacts["errors"]:
        for err in artifacts["errors"]:
            st.error(err)
        st.info("Place les fichiers `.joblib` nécessaires dans le dossier `models/`.")

    st.sidebar.header("⚙️ Configuration")
    model_choice = st.sidebar.selectbox("Choose the prediction model", ["SVC (Support Vector Machine)", "XGBoost"])
    page = st.sidebar.radio("Navigation", ["Prediction", "Dashboard"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("**SVC** : Accuracy 0.974, Precision 1.00, Recall 0.929\n\n**XGBoost** : Accuracy 0.974, Precision 1.00, Recall 0.929, F1-score 0.963, ROC-AUC 0.996")

    if page == "Prediction":
        st.subheader("📋 Tissue characteristics")
        st.caption("Adjust the values below or keep the defaults based on dataset averages.")
        input_values = {}
        tab_mean, tab_se, tab_worst = st.tabs(["Mean", "Standard Error (se)", "Worst"])
        mean_features = [f for f in FEATURES if f.endswith("_mean")]
        se_features = [f for f in FEATURES if f.endswith("_se")]
        worst_features = [f for f in FEATURES if f.endswith("_worst")]

        for tab, feats in [(tab_mean, mean_features), (tab_se, se_features), (tab_worst, worst_features)]:
            with tab:
                cols = st.columns(3)
                for i, feat in enumerate(feats):
                    with cols[i % 3]:
                        input_values[feat] = st.number_input(feat, value=float(DEFAULT_VALUES[feat]), format="%.5f", key=feat)

        if st.button("🔍 Predict Diagnosis", type="primary", use_container_width=True):
            input_df = pd.DataFrame([input_values])[FEATURES]
            label, proba_benign = predict(model_choice, artifacts, input_df)
            if label is None:
                st.error("Unable to make a prediction: missing model or scaler.")
            else:
                proba_malignant = 1 - proba_benign
                c1, c2 = st.columns(2)
                with c1:
                    if "Malignant" in label:
                        st.error(f"### Result: {label}")
                    else:
                        st.success(f"### Result: {label}")
                with c2:
                    st.metric("Probability of malignancy", f"{proba_malignant * 100:.2f}%")
                    st.progress(float(min(max(proba_malignant, 0.0), 1.0)))
                st.caption(f"Model used: **{model_choice}**. This prediction is for educational purposes only.")

    else:
        st.subheader("📊 Dataset dashboard")
        total_cases = len(df)
        benign_pct = (df["diagnosis"].eq("Benign").mean()) * 100
        malignant_pct = (df["diagnosis"].eq("Malignant").mean()) * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Total cases", total_cases)
        col2.metric("Benign", f"{benign_pct:.1f}%")
        col3.metric("Malignant", f"{malignant_pct:.1f}%")

        dash1, dash2, dash3, dash4 = st.tabs(["Overview", "EDA", "Feature insights", "Interpretability"])

        with dash1:
            a, b = st.columns([1, 1])
            with a:
                st.pyplot(plot_class_distribution(df), use_container_width=True)
            with b:
                st.pyplot(plot_mean_comparison(df), use_container_width=True)

        with dash2:
            full_matrix = st.toggle("Show full correlation matrix", value=False)
            st.pyplot(plot_correlation_heatmap(df, full_matrix=full_matrix), use_container_width=True)
            scaled_view = st.toggle("Use scaled feature distribution", value=False)
            st.pyplot(plot_feature_distribution(df, scaled=scaled_view), use_container_width=True)

        with dash3:
            st.markdown("### Key observations")
            st.markdown("- Radius, perimeter and area are strongly correlated.")
            st.markdown("- Malignant cases tend to show larger radius, perimeter and area values.")
            st.markdown("- Automatic plots are generated inside the app from `sklearn.datasets.load_breast_cancer()`.")
            st.dataframe(df.head(10), use_container_width=True)

        with dash4:
            st.markdown("### Model interpretability")
            st.markdown(
                "This section shows feature importance for **XGBoost** "
                "and an imported visualization for **SVC**."
            )
        
            col_xgb, col_svc = st.columns(2)
        
            # XGBoost
            with col_xgb:
                st.markdown("#### XGBoost feature importance")
                fig_imp = plot_xgb_feature_importance(artifacts, top_n=15)
                if fig_imp is not None:
                    st.pyplot(fig_imp, use_container_width=True)
                else:
                    st.info(
                        "XGBoost model or its feature list is missing, "
                        "so importance cannot be computed."
                    )
        
            # SVC (image uploadée depuis le notebook)
            with col_svc:
                st.markdown("#### SVC feature importance (notebook)")
                #st.markdown(
                #    "The SVC importance plot is generated in the training notebook "
                #    "using permutation importance and uploaded here."
                #)
        
                svc_image_path = "svc_feature_importance.png"
                if os.path.exists(svc_image_path):
                    st.image(
                        svc_image_path,
                        caption="SVC Permutation Feature Importance",
                        use_column_width=True,
                    )
                else:
                    st.info(
                        "Upload `svc_feature_importance.png` into the `models/` folder "
                        "to display the SVC importance plot."
                    )


    st.markdown("---")
    st.caption("Dataset: Breast Cancer Wisconsin (Diagnostic) loaded directly from scikit-learn.")

if __name__ == "__main__":
    main()
