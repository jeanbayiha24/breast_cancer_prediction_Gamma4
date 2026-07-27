# Breast Cancer Diagnosis Prediction – Gamma G4

This repository contains the code and assets for our **Breast Cancer Diagnosis Prediction** project based on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. The goal is to predict whether a breast mass is **benign (B)** or **malignant (M)** from FNA (Fine Needle Aspirate) image features and to compare several machine learning models, then deploy the best ones as an interactive web app.

Our Streamlit web application is available at:

> https://breastcancerpredictiongamma4.streamlit.app/

***

## 1. Project Overview

### Objective

- Predict tumor **diagnosis** (benign vs malignant) using 30 numerical features describing cell nuclei.
- Compare multiple classifiers:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Classifier (SVC)
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Analyze feature importance and the impact of **multicollinearity-aware feature selection**.
- Deploy the two best-performing models (SVC and XGBoost) in a simple clinical-style interface.

### Clinical framing

A **false negative** (predicting benign when the tumor is actually malignant) is much more costly than a false positive. For model comparison we therefore pay particular attention to:

- **Recall** on the malignant class (sensitivity)
- **F1-score**
- **ROC-AUC**, in addition to accuracy

***

## 2. Data & Preprocessing

- Dataset: **Breast Cancer Wisconsin (Diagnostic)** – 569 samples, 30 numerical features + diagnosis (B/M).
- Basic steps:
  - Drop technical columns (`id`, unnamed columns).
  - Encode `diagnosis` as binary target: `M → 1`, `B → 0`.
  - Train/test split with stratification (80/20).
  - **Standardization** (`StandardScaler`) for models that are sensitive to feature scales (Logistic Regression, SVC, KNN) and for clean cross-model comparison.

### Handling multicollinearity

The original 30 features include many highly correlated variables (e.g. `radius_mean`, `perimeter_mean`, `area_mean`, and their `_worst` / `_se` versions). To avoid redundant information and unstable coefficients:

1. Compute the full correlation matrix.
2. For each pair with $$|r| > 0.90$$, drop the feature that is **less correlated with the target**.
3. This yields a reduced set of **20 informative, less collinear features**, used in the XGBoost pipeline and in the separate Gamma G4 report.

We also keep a version of the pipeline using all 30 features to compare the impact of this feature-selection step.

***

## 3. Modeling

### Models evaluated in `untitled1_1.py`

Using the common helper function `evaluate_model()`, we train and evaluate:

- `LogisticRegression`
- `KNeighborsClassifier`
- `SVC` (RBF kernel, `probability=True`)
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `XGBClassifier`

Metrics computed on the held-out test set (20%):

- Accuracy
- Precision (positive class = malignant)
- Recall (sensitivity)
- F1-score
- ROC-AUC

### Key result

On our dataset, both **SVC** and **XGBoost** reach nearly identical performance (Accuracy ≈ **0.974**, Precision = **1.00**, Recall ≈ **0.93**, F1 ≈ **0.963**), despite the approach chosen, with only marginal differences in ROC-AUC. This suggests that, once proper feature selection and preprocessing are applied, multiple high-capacity models saturate at the same performance ceiling on this task.

Because of the clinical priority on recall for malignant cases and the excellent overall performance, we decided to **deploy both SVC and XGBoost** in the web application.

***

## 4. Streamlit Web Application

The interactive app is implemented in **`app.py`** and served via Streamlit Cloud.

### Main features

- **Model selector** in the sidebar:
  - `SVC (Support Vector Machine)`
  - `XGBoost`
- Form organized into three tabs for the 30 features:
  - `Mean`
  - `Standard Error (se)`
  - `Worst`
- Each input is pre-filled with approximate dataset means for convenience.
- For each model, the app displays:
  - **Predicted class**: Benign (B) or Malignant (M)
  - **Probability of malignancy** (in %)
  - A progress bar visualizing the malignancy probability

### Under the hood

- **SVC pipeline**
  - Trained on the 30 original features.
  - Uses a `StandardScaler` fitted on the training set.
  - Inference: user inputs → dataframe → `scaler.transform()` → `svc.predict()` / `svc.predict_proba()`.

- **XGBoost pipeline**
  - Trained on **20 selected features** (reduced multicollinearity).
  - Uses its own scaler (`xgb_scaler.joblib`) fitted on these 20 features.
  - During inference, the app:
    1. Subsets the user inputs to the 20 selected feature names (`xgb_selected_features.joblib`).
    2. Applies the XGBoost-specific scaler.
    3. Calls `xgb.predict()` and `xgb.predict_proba()`.

Even when they use different feature sets and preprocessing, SVC and XGBoost now agree on the predicted class for the same patient profile; remaining differences appear only in the probability values (which is expected given different calibration methods).

***

## 5. Repository Structure

Example structure :

```text
.
├── app.py                        # Streamlit app (SVC + XGBoost)
├── models/                       # Saved models and preprocessors
│   ├── svc_breast_cancer_model.joblib
│   ├── scaler.joblib             # Scaler for SVC (30 features)
│   ├── xgboost_best_model.joblib
│   ├── xgb_scaler.joblib         # Scaler for XGBoost (20 features)
│   └── xgb_selected_features.joblib
├── README.md              # This README (for GitHub)
└── requirements.txt              # Python dependencies
```

***

## 6. Installation & Local Run

### 6.1. Clone the repository

```bash
git clone https://github.com/jeanbayiha24/breast_cancer_prediction_Gamma4.git
cd breast_cancer_prediction_Gamma4-main
```

### 6.2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
```

### 6.3. Install dependencies

```bash
pip install -r requirements.txt
```

### 6.4. Place model files

Make sure the following files are present in the `models/` directory:

- `svc_breast_cancer_model.joblib`
- `scaler.joblib` (for the SVC)
- `xgboost_best_model.joblib`
- `xgb_scaler.joblib`
- `xgb_selected_features.joblib`

### 6.5. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser (default: http://localhost:8501).

***

## 7. Limitations & Next Steps

- The WDBC dataset is relatively small (569 samples), and performance is measured on a single train/test split. External validation on truly independent clinical data would be required before any real-world use.
- The app does **not** replace a medical diagnosis; it is a didactic prototype to illustrate how ML models can assist in risk estimation.
- Potential extensions:
  - Calibrating probabilities (e.g. Platt scaling / isotonic regression) and comparing probability calibration across models.
  - Adding uncertainty estimates or confidence intervals.
  - Extending to additional datasets and features (e.g. imaging or genomic data).

***

## 8. Credits

This project was developed as part of the **Gamma G4** capstone, by the team members of the IIP Program 2026. The Streamlit app and codebase were designed to make the modeling pipeline transparent and reproducible for teaching and demonstration purposes.
