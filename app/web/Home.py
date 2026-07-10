from __future__ import annotations

import pandas as pd
import streamlit as st

from app.web.api_client import ApiClientError, get_past_predictions
from app.web.api_client import predict_demand


st.set_page_config(
    page_title="Bike Demand Predictor",
    page_icon=":bike:",
    layout="wide",
)


def build_feature_payload() -> dict:
    left, middle, right = st.columns(3)

    with left:
        season = st.selectbox(
            "Season",
            options=[1, 2, 3, 4],
            format_func=lambda value: {
                1: "Spring",
                2: "Summer",
                3: "Fall",
                4: "Winter",
            }[value],
        )
        mnth = st.slider("Month", min_value=1, max_value=12, value=6)
        hr = st.slider("Hour", min_value=0, max_value=23, value=8)
        weekday = st.slider("Weekday", min_value=0, max_value=6, value=1)

    with middle:
        yr = st.selectbox("Year index", options=[0, 1], index=1)
        holiday = st.selectbox("Holiday", options=[0, 1])
        workingday = st.selectbox("Working day", options=[0, 1], index=1)
        weathersit = st.selectbox(
            "Weather",
            options=[1, 2, 3, 4],
            format_func=lambda value: {
                1: "Clear",
                2: "Mist or cloudy",
                3: "Light rain or snow",
                4: "Heavy rain or snow",
            }[value],
        )

    with right:
        temp = st.slider("Temperature", 0.0, 1.0, 0.55, 0.01)
        atemp = st.slider("Feels-like temperature", 0.0, 1.0, 0.52, 0.01)
        hum = st.slider("Humidity", 0.0, 1.0, 0.60, 0.01)
        windspeed = st.slider("Wind speed", 0.0, 1.0, 0.20, 0.01)

    return {
        "season": season,
        "yr": yr,
        "mnth": mnth,
        "hr": hr,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
    }


def render_prediction_result(result: dict) -> None:
    predictions = result.get("predictions", [])
    if not predictions:
        st.warning("The API returned no prediction.")
        return

    predicted_count = predictions[0]["prediction"]
    st.metric("Predicted bike demand", f"{predicted_count:.0f} rentals")


def render_prediction_history() -> None:
    st.subheader("Past predictions")
    try:
        rows = get_past_predictions(limit=20)
    except ApiClientError as exc:
        st.info(str(exc))
        return

    if not rows:
        st.info("No predictions saved yet.")
        return

    table_rows = []
    for row in rows:
        features = row.get("features", {})
        table_rows.append(
            {
                "created_at": row.get("created_at"),
                "prediction": round(row.get("prediction", 0), 2),
                "hour": features.get("hr"),
                "month": features.get("mnth"),
                "weather": features.get("weathersit"),
                "temperature": features.get("temp"),
                "humidity": features.get("hum"),
                "source": row.get("source"),
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)


def main() -> None:
    st.title("Hourly Bike Sharing Demand Prediction")
    st.caption("FastAPI prediction demo with saved prediction history")

    features = build_feature_payload()

    if st.button("Predict demand", type="primary"):
        try:
            result = predict_demand(features)
        except ApiClientError as exc:
            st.error(str(exc))
        else:
            render_prediction_result(result)

    st.divider()
    render_prediction_history()


if __name__ == "__main__":
    main()
