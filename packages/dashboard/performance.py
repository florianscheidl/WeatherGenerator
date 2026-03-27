import logging
from datetime import datetime, timezone

import plotly.express as px
import polars as pl
import streamlit as st

from weathergen.dashboard.metrics import ST_TTL_SEC, get_experiment_id, setup_mflow

_logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
_logger.info("Setting up MLFlow")
client = setup_mflow()

# Hardcoded run_id for the only run with performance metrics so far.
PERF_RUN_ID = "c8mu32h9"

THROUGHPUT_KEY = "performance.throughput.mb_per_sec"

st.markdown(
    """
# Performance metrics

Throughput and utilization metrics logged under `performance.*`.
"""
)


@st.cache_data(ttl=ST_TTL_SEC)
def get_performance_metrics(wg_run_id: str) -> pl.DataFrame:
    """Fetch all performance.* metric time series for a given WG run_id."""
    runs = client.search_runs(
        experiment_ids=[get_experiment_id()],
        filter_string=f"tags.run_id = '{wg_run_id}'",
        max_results=1,
    )
    if not runs:
        return pl.DataFrame()

    run = runs[0]
    mlflow_run_id = run.info.run_id
    start_time_ms = run.info.start_time  # milliseconds since epoch

    # Discover which performance.* metrics were logged for this run
    perf_keys = [k for k in run.data.metrics if k.startswith("performance.")]
    if not perf_keys:
        return pl.DataFrame()

    # Fetch full history for each metric
    records = []
    for metric_key in perf_keys:
        history = client.get_metric_history(run_id=mlflow_run_id, key=metric_key)
        for m in history:
            records.append({"metric": metric_key, "step": m.step, "value": m.value})

    start_date = datetime.fromtimestamp(start_time_ms / 1000, tz=timezone.utc).date()
    return pl.DataFrame(records).with_columns(
        pl.lit(wg_run_id).alias("run_id"),
        pl.lit(start_date).alias("start_date"),
    )


perf_df = get_performance_metrics(PERF_RUN_ID)

if perf_df.is_empty():
    st.warning(f"No performance metrics found for run `{PERF_RUN_ID}`.")
else:
    throughput = perf_df.filter(pl.col("metric") == THROUGHPUT_KEY)

    if throughput.is_empty():
        st.warning(f"No `{THROUGHPUT_KEY}` entries found.")
    else:
        # One point per run: (start_date, min MB/s)
        plot_df = (
            throughput.group_by("run_id", "start_date")
            .agg(pl.col("value").min().alias("min_mb_per_sec"))
            .sort("start_date")
        )

        fig = px.scatter(
            plot_df.to_pandas(),
            x="start_date",
            y="min_mb_per_sec",
            hover_data={"run_id": True, "start_date": False},
            labels={"start_date": "", "min_mb_per_sec": "Min MB/s"},
            title="Minimum throughput per run",
        )
        fig.update_xaxes(tickformat="%Y-%m-%d")
        fig.update_traces(marker={"size": 10})
        st.plotly_chart(fig, use_container_width=True)
