
import streamlit as st
import pandas as pd
import joblib
from datetime import date
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌿",
    layout="wide"
)

# --------------------------------------------------
# Load trained models
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "aqi_models.joblib"

try:
    bundle = joblib.load(MODEL_PATH)

    reg_model = bundle["regression_model"]
    reg_name = bundle["regression_model_name"]

    clf_model = bundle["classification_model"]
    clf_name = bundle["classification_model_name"]

    cities = bundle["cities"]

except FileNotFoundError:
    st.error(
        "Model file 'aqi_models.joblib' was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()

except Exception as exc:
    st.error(f"Unable to load the trained models: {exc}")
    st.stop()


# --------------------------------------------------
# Season feature
# --------------------------------------------------
def season_from_month(month):
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Fall"


# --------------------------------------------------
# AQI category descriptions
# --------------------------------------------------
def category_description(category):
    descriptions = {
        "Good": (
            "Air quality is satisfactory, and air pollution "
            "poses little or no risk."
        ),
        "Moderate": (
            "Air quality is acceptable; unusually sensitive "
            "people may experience some concern."
        ),
        "Unhealthy (Sensitive)": (
            "Sensitive groups may experience health effects."
        ),
        "Unhealthy": (
            "Some members of the general public may experience "
            "health effects."
        ),
        "Very Unhealthy": (
            "Health alert: the risk of health effects is "
            "increased for everyone."
        ),
        "Hazardous": (
            "Health warning of emergency conditions; everyone "
            "is more likely to be affected."
        )
    }

    return descriptions.get(
        str(category),
        "Refer to your local air-quality authority for guidance."
    )


# --------------------------------------------------
# App title and introduction
# --------------------------------------------------
st.title("🌿 Air Quality Index (AQI) Predictor")

st.write(
    "A machine-learning application that predicts numerical "
    "AQI using regression and predicts the air-quality category "
    "using classification, based on pollutant and weather measurements."
)

st.caption("Mini Project · Phase 3 · Streamlit deployment")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("About this app")

    st.write(
        "This application uses a trained regression model "
        "to estimate numerical AQI and a trained classification "
        "model to predict the AQI category."
    )

    st.write(f"**Regression model:** {reg_name}")
    st.write(f"**Classification model:** {clf_name}")

    st.divider()

    st.caption(
        "Predictions depend on the dataset and model quality. "
        "They are estimates, not official monitoring readings "
        "or medical advice."
    )


# --------------------------------------------------
# User input fields
# --------------------------------------------------
st.subheader("Enter air-quality and weather measurements")

left, right = st.columns(2)

with left:
    city = st.selectbox(
        "City",
        options=cities,
        index=0
    )

    selected_date = st.date_input(
        "Observation date",
        value=date.today()
    )

    pm25 = st.number_input(
        "PM2.5",
        min_value=0.0,
        max_value=1000.0,
        value=28.0,
        step=1.0
    )

    pm10 = st.number_input(
        "PM10",
        min_value=0.0,
        max_value=1500.0,
        value=45.0,
        step=1.0
    )

    ozone = st.number_input(
        "Ozone",
        min_value=0.0,
        max_value=1000.0,
        value=30.0,
        step=1.0
    )

    no2 = st.number_input(
        "NO2",
        min_value=0.0,
        max_value=1000.0,
        value=20.0,
        step=1.0
    )


with right:
    co = st.number_input(
        "CO",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.1
    )

    so2 = st.number_input(
        "SO2",
        min_value=0.0,
        max_value=1000.0,
        value=10.0,
        step=1.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-50.0,
        max_value=60.0,
        value=22.0,
        step=0.5
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=55.0,
        step=1.0
    )


# --------------------------------------------------
# Prepare input features
# --------------------------------------------------
input_row = pd.DataFrame([{
    "PM2.5": pm25,
    "PM10": pm10,
    "Ozone": ozone,
    "NO2": no2,
    "CO": co,
    "SO2": so2,
    "Temperature": temperature,
    "Humidity": humidity,
    "Month": selected_date.month,
    "DayOfWeek": selected_date.weekday(),
    "City": city,
    "Season": season_from_month(selected_date.month)
}])


# --------------------------------------------------
# Predict AQI
# --------------------------------------------------
if st.button(
    "Predict AQI",
    type="primary",
    use_container_width=True
):
    try:
        # Predict numerical AQI using regression
        predicted_aqi = float(
            reg_model.predict(input_row)[0]
        )

        # Prevent a negative numerical AQI estimate
        predicted_aqi = max(0.0, predicted_aqi)

        # Predict category independently using classification
        predicted_category = str(
            clf_model.predict(input_row)[0]
        )

        # Display prediction results
        st.divider()
        st.subheader("Prediction Results")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Estimated AQI",
                f"{predicted_aqi:.1f}"
            )
            st.caption(f"Predicted by {reg_name}")

        with c2:
            st.metric(
                "Predicted AQI Category",
                predicted_category
            )
            st.caption(f"Predicted by {clf_name}")

        st.caption(
            "The numerical AQI is predicted using regression, "
            "while the AQI category is predicted independently "
            "using classification. The two predictions may differ."
        )

        # Display category explanation
        st.info(
            category_description(predicted_category)
        )

        # Display user input summary
        st.write("**Input Summary**")

        st.dataframe(
            input_row,
            use_container_width=True,
            hide_index=True
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# --------------------------------------------------
# Phase 2 evaluation metrics
# --------------------------------------------------
with st.expander("Phase 2 Evaluation Snapshot"):

    metrics = bundle.get("phase2_metrics", {})

    reg_metrics = pd.DataFrame(
        metrics.get("regression", [])
    )

    clf_metrics = pd.DataFrame(
        metrics.get("classification", [])
    )

    st.write("**Regression comparison (held-out test set)**")

    if not reg_metrics.empty:
        st.dataframe(
            reg_metrics.round(4),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write(
            "No regression evaluation metrics available."
        )

    st.write("**Classification comparison (held-out test set)**")

    if not clf_metrics.empty:
        st.dataframe(
            clf_metrics.round(4),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write(
            "No classification evaluation metrics available."
        )

    st.caption(
        "The regression and classification models were compared "
        "using held-out test-set metrics. The live application "
        "uses both saved models for independent predictions."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Academic demonstration only. AQI estimates and category "
    "labels should not replace readings from official "
    "environmental monitoring agencies."
)
