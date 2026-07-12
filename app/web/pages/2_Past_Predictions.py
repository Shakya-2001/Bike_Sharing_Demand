from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from app.web.api_client import ApiClientError, get_past_predictions
from app.web.prediction_data import SOURCE_NAMES, history_frame


st.set_page_config(
    page_title="Past Predictions",
    page_icon=":bike:",
    layout="wide",
)


def main() -> None:
    st.title("Past Predictions")

    today = date.today()
    left, middle, right, last = st.columns(4)
    with left:
        start_date = st.date_input(
            "Start date",
            value=today - timedelta(days=30),
        )
    with middle:
        end_date = st.date_input("End date", value=today)
    with right:
        source = st.selectbox(
            "Source",
            options=list(SOURCE_NAMES),
            format_func=lambda value: SOURCE_NAMES[value],
        )
    with last:
        limit = st.number_input("Maximum results", 1, 1000, 100)

    if start_date > end_date:
        st.error("Start date must not be after end date.")
        return

    try:
        rows = get_past_predictions(
            source=source,
            start_date=start_date,
            end_date=end_date,
            limit=int(limit),
        )
    except ApiClientError as exc:
        st.error(str(exc))
        return

    frame = history_frame(rows)
    if frame.empty:
        st.info("No predictions found for the selected filters.")
        return
    st.dataframe(frame, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
