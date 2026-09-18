"""Simple dashboard-chart data helpers."""

import pandas as pd


def events_dataframe(events: list[dict]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=["frame", "event_type", "object_id", "message"])
    return pd.DataFrame(events)
