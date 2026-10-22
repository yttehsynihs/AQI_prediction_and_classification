
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
bundle = joblib.load(MODEL_PATH)

reg_model = bundle["regression_model"]
reg_name = bundle["regression_model_name"]

clf_model = bundle["classification_model"]
clf_name = bundle["classification_model_name"]

cities = bundle["cities"]

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
# Calculate AQI category from numerical AQI
# US-style AQI thresholds
# --------------------------------------------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# --------------------------------------------------
# Category descriptions
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
        category,
        "Refer to your local air-quality authority for guidance."
    )


# --------------------------------------------------
# App title and introduction
# --------------------------------------------------
st.title("🌿 Air Quality Index (AQI) Predictor")

st.write(
    "A machine-learning application for estimating numerical "
    "AQI and calculating its air-quality category from pollutant "
    "and weather measurements."
)

st.caption("Mini Project · Phase 3 · Streamlit deployment")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("About this app")

    st.write(
        "This application uses a trained machine-learning "
        "regression model to estimate AQI. The category is "
        "calculated directly from the estimated numerical AQI."
    )

    st.write(f"**Regression model:** {reg_name}")

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
# Predict numerical AQI and calculate category
# --------------------------------------------------
if st.button(
    "Predict AQI",
    type="primary",
    use_container_width=True
):
    try:
        # Predict numerical AQI
        predicted_aqi = float(
            reg_model.predict(input_row)[0]
        )

        # Prevent negative AQI predictions
        predicted_aqi = max(0.0, predicted_aqi)

        # Calculate category directly from numerical AQI
        predicted_category = get_aqi_category(predicted_aqi)

        # Display results
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
                "AQI Category",
                predicted_category
            )

            st.caption("Calculated from estimated AQI")

        # Display explanation
        st.info(
            category_description(predicted_category)
        )

        st.caption(
            "The category is calculated from the numerical AQI "
            "using the configured US-style AQI thresholds. "
            "The result is an estimate, not an official reading."
        )

        # Display user inputs
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
        st.write("No regression evaluation metrics available.")

    st.write("**Classification comparison (held-out test set)**")

    if not clf_metrics.empty:
        st.dataframe(
            clf_metrics.round(4),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No classification evaluation metrics available.")

    st.caption(
        "The classification metrics are retained for your Phase 2 "
        "model comparison. The live AQI category above is now "
        "calculated from the regression prediction."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Academic demonstration only. AQI values and category labels "
    "should not replace readings from official environmental "
    "monitoring agencies."
)
