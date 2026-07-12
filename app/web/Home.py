from __future__ import annotations

import pandas as pd
import streamlit as st

from app.web.api_client import ApiClientError, predict_demand
from app.web.prediction_data import MONTH_NAMES, SEASON_MONTHS, WEEKDAY_NAMES
from app.web.prediction_data import batch_rows, prediction_frame
from app.web.prediction_data import normalize_feels_like_temperature
from app.web.prediction_data import normalize_humidity, normalize_temperature
from app.web.prediction_data import normalize_wind_speed


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
        mnth = st.selectbox(
            "Month",
            options=SEASON_MONTHS[season],
            format_func=lambda value: MONTH_NAMES[value],
        )
        hr = st.slider("Hour", min_value=0, max_value=23, value=8)
        weekday = st.selectbox(
            "Weekday",
            options=list(WEEKDAY_NAMES),
            index=1,
            format_func=lambda value: WEEKDAY_NAMES[value],
        )

    with middle:
        year = st.selectbox("Year", options=[2011, 2012], index=1)
        holiday = st.selectbox(
            "Holiday",
            options=[0, 1],
            format_func=lambda value: "Yes" if value else "No",
        )
        workingday = st.selectbox(
            "Working day",
            options=[0, 1],
            index=1,
            format_func=lambda value: "Yes" if value else "No",
        )
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
        temperature_c = st.slider(
            "Temperature (°C)",
            min_value=-8.0,
            max_value=39.0,
            value=18.0,
            step=0.5,
        )
        feels_like_c = st.slider(
            "Feels-like temperature (°C)",
            min_value=-16.0,
            max_value=50.0,
            value=18.0,
            step=0.5,
        )
        humidity_percent = st.slider(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=60,
        )
        wind_speed_kmh = st.slider(
            "Wind speed (km/h)",
            min_value=0.0,
            max_value=67.0,
            value=13.5,
            step=0.5,
        )

    yr = year - 2011
    temp = normalize_temperature(temperature_c)
    atemp = normalize_feels_like_temperature(feels_like_c)
    hum = normalize_humidity(humidity_percent)
    windspeed = normalize_wind_speed(wind_speed_kmh)

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


def show_result(result: dict, batch: bool = False) -> None:
    frame = prediction_frame(result)
    if frame.empty:
        st.warning("The API returned no predictions.")
        return

    predictions = frame["prediction"].clip(lower=0)
    if batch:
        left, middle, right = st.columns(3)
        left.metric("Predicted rows", len(frame))
        middle.metric(
            "Average hourly demand",
            f"{predictions.mean():,.0f} rentals",
        )
        right.metric(
            "Total predicted rentals",
            f"{predictions.sum():,.0f}",
        )
    else:
        st.metric(
            "Estimated bike rentals",
            f"{predictions.iloc[0]:,.0f}",
        )

    st.dataframe(frame, width="stretch", hide_index=True)


def render_single_prediction() -> None:
    features = build_feature_payload()
    if st.button("Predict demand", type="primary"):
        try:
            show_result(predict_demand(features))
        except ApiClientError as exc:
            st.error(str(exc))


def render_batch_prediction() -> None:
    uploaded = st.file_uploader("Inference CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        data = pd.read_csv(uploaded)
        rows = batch_rows(data)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(str(exc))
        return

    st.dataframe(data, width="stretch", hide_index=True)
    if st.button("Predict batch", type="primary"):
        try:
            show_result(predict_demand(rows), batch=True)
        except ApiClientError as exc:
            st.error(str(exc))


def main() -> None:
    st.title("Hourly Bike Sharing Demand Prediction")
    single_tab, batch_tab = st.tabs(["Single prediction", "Batch prediction"])
    with single_tab:
        render_single_prediction()
    with batch_tab:
        render_batch_prediction()


if __name__ == "__main__":
    navigation = st.navigation(
        [
            st.Page(main, title="Prediction", default=True),
            st.Page(
                "pages/2_Past_Predictions.py",
                title="Past Predictions",
            ),
        ]
    )
    navigation.run()
