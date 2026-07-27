# breast_cancer_prediction_Gamma4
Capstone project realised by the Group Gamma 4 during the IIP 2026 Program of iiAfrica.

## Structure
breast_cancer_app/
├── app.py
├── requirements.txt
└── models/
    ├── svc_breast_cancer_model.joblib
    ├── xgboost_best_model.joblib
    └── scaler.joblib

## Lancer l'app
1. pip install -r requirements.txt
2. streamlit run app.py

## Notes
- Le SVC nécessite le scaler (features standardisées).
- Le XGBoost utilise les features brutes (non standardisées), comme dans le notebook d'entraînement.
