# AQI Streamlit Deployment — Phase 3

## What is included
- `app.py`: Streamlit interface for AQI prediction.
- `aqi_models.joblib`: saved pipelines for the best regression and classification models.
- `requirements.txt`: required Python packages.

## Models selected from Phase 2 test-set evaluation
- Regression: **Random Forest Regressor (Ensemble)** (lowest RMSE among the three Phase 2 regression models).
- Classification: **Random Forest Classifier (Ensemble)** (highest macro-F1 among the three Phase 2 classification models).

Test-set snapshot:
- Regression metrics: [{"Model": "Random Forest Regressor (Ensemble)", "MAE": 8.246799628942485, "RMSE": 17.42206175800713, "R2": 0.5447651551566517}, {"Model": "KNN Regressor", "MAE": 10.715213358070502, "RMSE": 19.08461482951895, "R2": 0.45373522386866627}, {"Model": "Multiple Linear Regression", "MAE": 10.441924733991286, "RMSE": 19.58624371097239, "R2": 0.4246412699842266}]
- Classification metrics: [{"Model": "Logistic Regression", "Accuracy": 0.637291280148423, "Macro-F1": 0.38763360195991026, "Weighted-F1": 0.6769108557250136}, {"Model": "Decision Tree Classifier", "Accuracy": 0.7699443413729128, "Macro-F1": 0.39248620532390677, "Weighted-F1": 0.7688550370014225}, {"Model": "Random Forest Classifier (Ensemble)", "Accuracy": 0.8209647495361782, "Macro-F1": 0.43292077822655156, "Weighted-F1": 0.815565579133444}]

The saved models are then refitted on all cleaned Phase 1 rows for deployment after model selection.

## Run locally (Windows)
1. Install Python 3.10+.
2. Extract this folder.
3. Open Command Prompt or terminal in this folder.
4. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   python -m streamlit run app.py
   ```
6. Streamlit will open the local app in your browser, usually at `http://localhost:8501`.

## Deploy online (optional)
Push these files to a GitHub repository and create an app in Streamlit Community Cloud, selecting `app.py` as the entry point.

## Important limitation
The dataset has highly imbalanced AQI categories (especially Good and Hazardous, with very few examples). Classification results for rare categories may be unreliable. This is an academic demonstration and should not replace official AQI readings.
