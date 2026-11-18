"""
Streamlit dashboard for the natural gas weather-signal research pipeline.

Run with: streamlit run app.py
All figures are read from the artifacts written by `python main.py`; nothing
displayed here is hard-coded.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path("data")

st.set_page_config(page_title="Natural Gas Weather-Signal Research", layout="wide")

st.title("Weather-Driven Signal Research: Henry Hub Natural Gas")
st.caption(
    "Research prototype. Out-of-sample metrics come from an expanding-window walk-forward "
    "evaluation. Nothing here is investment advice."
)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def load_parquet(path: str, mtime: float) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_csv(path: str, mtime: float, **kwargs) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p, **kwargs) if p.exists() else None


@st.cache_data
def load_json(path: str, mtime: float) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _get(name: str, loader, **kwargs):
    path = DATA_DIR / name
    return loader(str(path), _mtime(path), **kwargs)


def fmt_pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None or x != x else f"{x:.{digits}%}"


def fmt_num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None or x != x else f"{x:.{digits}f}"


df = _get("final_dataset.parquet", load_parquet)
if df is None:
    st.error("`data/final_dataset.parquet` not found. Run `python main.py` first.")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
metrics = _get("metrics.json", load_json)
fold_metrics = _get("fold_metrics.csv", load_csv)
bt_metrics = _get("backtest_metrics.json", load_json)
bt_daily = _get("backtest_daily.parquet", load_parquet)
importance = _get("feature_importance.csv", load_csv, index_col=0)

tab_data, tab_eval, tab_bt, tab_feat = st.tabs(
    ["Data", "Walk-forward evaluation", "Backtest", "Feature importance"]
)

with tab_data:
    last = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last date", f"{last['date']:%Y-%m-%d}")
    c2.metric("Last close (USD/MMBtu)", f"{last['close']:.3f}")
    c3.metric("National avg HDD", f"{last['national_avg_hdd']:.1f}")
    c4.metric("HDD anomaly vs prior-year norm", f"{last['national_hdd_anomaly']:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Front-month close", yaxis="y1"))
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["national_avg_hdd"],
            name="National avg HDD",
            yaxis="y2",
            line=dict(color="orange", width=1),
            opacity=0.6,
        )
    )
    fig.update_layout(
        title="Front-month price and national heating degree days",
        yaxis=dict(title="USD/MMBtu"),
        yaxis2=dict(title="HDD", overlaying="y", side="right"),
        template="plotly_dark",
        height=450,
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        df[["date", "close", "national_avg_hdd", "national_hdd_anomaly", "volume"]].tail(10),
        width="stretch",
    )

with tab_eval:
    if metrics is None or fold_metrics is None:
        st.info("Run `python main.py` to generate walk-forward results.")
    else:
        a, b, c, d = st.columns(4)
        a.metric(
            "Directional accuracy (pooled OOS)", fmt_pct(metrics["pooled_directional_accuracy"])
        )
        b.metric("Naive majority baseline", fmt_pct(metrics["naive_majority_accuracy"]))
        c.metric("p-value vs naive baseline", fmt_num(metrics["pvalue_vs_naive_majority"], 3))
        d.metric("Information coefficient", fmt_num(metrics["pooled_information_coefficient"], 3))

        st.markdown(
            f"**{metrics['n_oos_observations']}** out-of-sample days across "
            f"**{metrics['n_folds']}** yearly folds "
            f"({metrics['oos_period'][0]} to {metrics['oos_period'][1]}). "
            f"Fold accuracy: {fmt_pct(metrics['fold_accuracy_mean'])} "
            f"+/- {fmt_pct(metrics['fold_accuracy_std'])} (std). "
            f"p-value vs coin flip: {fmt_num(metrics['pvalue_vs_coin_flip'], 3)}."
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=fold_metrics["test_year"].astype(str),
                y=fold_metrics["directional_accuracy"],
                name="Model",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fold_metrics["test_year"].astype(str),
                y=fold_metrics["naive_accuracy"],
                name="Naive majority",
                mode="lines+markers",
            )
        )
        fig.add_hline(y=0.5, line_dash="dot", line_color="grey")
        fig.update_layout(
            title="Directional accuracy by test year",
            yaxis=dict(tickformat=".0%", range=[0.35, 0.7]),
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            fold_metrics[
                [
                    "fold",
                    "test_year",
                    "n",
                    "directional_accuracy",
                    "naive_accuracy",
                    "information_coefficient",
                    "mae",
                    "mae_zero_forecast",
                ]
            ],
            width="stretch",
        )

with tab_bt:
    if bt_metrics is None or bt_daily is None:
        st.info("Run `python main.py` to generate the backtest.")
    else:
        bt_daily["date"] = pd.to_datetime(bt_daily["date"])
        a, b, c, d = st.columns(4)
        a.metric("Sharpe (net of costs)", fmt_num(bt_metrics["sharpe_net"]))
        b.metric("Sharpe (gross)", fmt_num(bt_metrics["sharpe_gross"]))
        c.metric("Sharpe (always-long benchmark)", fmt_num(bt_metrics["sharpe_benchmark_long"]))
        d.metric("Max drawdown (net)", fmt_pct(bt_metrics["max_drawdown_net"]))

        st.markdown(
            f"Volatility target {fmt_pct(bt_metrics['target_vol'], 0)}, slippage "
            f"{bt_metrics['slippage_ticks']} ticks per unit of turnover. "
            f"Sharpe standard error is roughly {fmt_num(bt_metrics['sharpe_net_approx_stderr'])}; "
            f"fold Sharpe: {fmt_num(bt_metrics['fold_sharpe_net_mean'])} "
            f"+/- {fmt_num(bt_metrics['fold_sharpe_net_std'])}."
        )

        fig = go.Figure()
        for col, label in [
            ("equity_net", "Strategy (net)"),
            ("equity_gross", "Strategy (gross)"),
            ("equity_benchmark", "Always long (vol-targeted)"),
        ]:
            fig.add_trace(go.Scatter(x=bt_daily["date"], y=bt_daily[col], name=label))
        fig.update_layout(
            title="Cumulative equity (out-of-sample)",
            template="plotly_dark",
            height=450,
        )
        st.plotly_chart(fig, width="stretch")

        fold_sharpe = pd.Series(bt_metrics["fold_sharpe_net"])
        fig2 = go.Figure(go.Bar(x=fold_sharpe.index.astype(str), y=fold_sharpe.values))
        fig2.update_layout(
            title="Net Sharpe by walk-forward fold",
            xaxis_title="Fold",
            template="plotly_dark",
            height=320,
        )
        st.plotly_chart(fig2, width="stretch")

with tab_feat:
    if importance is None:
        st.info("Run `python main.py` to compute feature importance.")
    else:
        ordered = importance["gain_share"].sort_values()
        fig = go.Figure(go.Bar(x=ordered.values, y=ordered.index, orientation="h"))
        fig.update_layout(
            title="Share of total gain, averaged over walk-forward folds",
            xaxis=dict(tickformat=".0%"),
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.caption("Antonio (AntonioOYM) | Quantitative climate-finance research project")
