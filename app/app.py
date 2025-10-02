import sys, os, json, yaml
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import streamlit as st
import altair as alt

# ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.lstm import LSTMClassifier
from src.utils import get_device


def load_cfg(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def infer_next_horizons(cfg, use_tuned_thresholds=True, default_threshold=0.50):
    """Return forecast df (date, prob, threshold, pred), tuned thresholds dict, processed test df, coverage window."""
    device = get_device()
    H = int(cfg["features"].get("target_horizons", 1))

    # model
    ckpt_path = cfg["train"]["ckpt_path"]
    ckpt = torch.load(ckpt_path, map_location=device)
    model = LSTMClassifier(
        input_size=ckpt["input_size"],
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        out_dim=H,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # processed data
    test_df = pd.read_parquet(cfg["data"]["processed_test"])
    target_cols = [f"heatwave_h{h}" for h in range(1, H + 1)]
    feature_cols = [c for c in test_df.columns if c not in target_cols]
    lookback = int(cfg["features"]["lookback_days"])

    # coverage window across splits if available
    try:
        train = pd.read_parquet(cfg["data"]["processed_train"])
        valid = pd.read_parquet(cfg["data"]["processed_valid"])
        coverage_start = min(train.index.min(), valid.index.min(), test_df.index.min())
        coverage_end = max(train.index.max(), valid.index.max(), test_df.index.max())
    except Exception:
        coverage_start, coverage_end = test_df.index.min(), test_df.index.max()

    # last sequence -> probabilities for H next days
    X = test_df[feature_cols].values.astype(np.float32)
    x_seq = torch.tensor(X[-lookback:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x_seq)).cpu().numpy().ravel()  # (H,)

    last_day = test_df.index[-1]
    dates = [last_day + timedelta(days=h) for h in range(1, H + 1)]

    # thresholds: tuned (if present) else fixed default
    tuned = {}
    ts = np.full(H, float(default_threshold), dtype=float)
    th_path = os.path.join(os.path.dirname(cfg["train"]["ckpt_path"]), "thresholds.json")
    if use_tuned_thresholds and os.path.exists(th_path):
        with open(th_path) as f:
            tuned = json.load(f)
        ts = np.array([float(tuned.get(f"h{i+1}", default_threshold)) for i in range(H)])

    labels = (probs >= ts).astype(int)

    forecast = pd.DataFrame(
        {"date": pd.to_datetime(dates), "p_heatwave": probs, "threshold": ts, "predicted_heatwave": labels}
    ).sort_values("date").reset_index(drop=True)

    return forecast, tuned, test_df, pd.to_datetime(coverage_start), pd.to_datetime(coverage_end)


def line_chart_dates(df, x_col, y_col, title=None, fmt="%b %d"):
    base = alt.Chart(df).mark_line().encode(
        x=alt.X(f"{x_col}:T", axis=alt.Axis(format=fmt, labelAngle=-30)),
        y=alt.Y(f"{y_col}:Q")
    )
    if title:
        base = base.properties(title=title)
    return base.interactive()


def group_heatwave_spells(forecast_df: pd.DataFrame):
    """Group consecutive predicted_heatwave==1 rows into spells and compute stats."""
    df = forecast_df.sort_values("date").reset_index(drop=True).copy()
    hw = df[df["predicted_heatwave"] == 1].copy()
    spells = []
    if hw.empty:
        return spells

    start = prev = hw.iloc[0]["date"]
    probs = [float(hw.iloc[0]["p_heatwave"])]
    ths = [float(hw.iloc[0]["threshold"])]

    for i in range(1, len(hw)):
        d = hw.iloc[i]["date"]
        if (d - prev).days == 1:
            # continue same spell
            prev = d
            probs.append(float(hw.iloc[i]["p_heatwave"]))
            ths.append(float(hw.iloc[i]["threshold"]))
        else:
            # close previous spell
            spells.append({
                "start": start,
                "end": prev,
                "days": (prev - start).days + 1,
                "min_prob": float(np.min(probs)),
                "mean_prob": float(np.mean(probs)),
                "min_threshold": float(np.min(ths))
            })
            # start new
            start = prev = d
            probs = [float(hw.iloc[i]["p_heatwave"])]
            ths = [float(hw.iloc[i]["threshold"])]

    # close last spell
    spells.append({
        "start": start,
        "end": prev,
        "days": (prev - start).days + 1,
        "min_prob": float(np.min(probs)),
        "mean_prob": float(np.mean(probs)),
        "min_threshold": float(np.min(ths))
    })
    return spells


def main():
    st.set_page_config(page_title="Delhi Heatwave Predictor", page_icon="🔥", layout="centered")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        cfg_path = st.text_input("Config file", value="configs/default.yaml")
        use_tuned = st.checkbox("Use tuned per-horizon thresholds (if available)", value=True)
        show_history_days = st.slider("Show recent history (days)", 30, 180, 90, 15)

    try:
        cfg = load_cfg(cfg_path)
    except Exception as e:
        st.error(f"Failed to load config at {cfg_path}: {e}")
        return

    st.title("🔥 Delhi Heatwave Predictor — 7-Day Outlook")

    forecast_df, tuned, test_df, cov_start, cov_end = infer_next_horizons(
        cfg, use_tuned_thresholds=use_tuned, default_threshold=0.50
    )

    # Data coverage
    st.subheader("Data Coverage")
    st.info(f"Using processed data for this area from **{cov_start.date()}** to **{cov_end.date()}**.")

    # Forecasted spells
    st.subheader("Forecast: Next Heatwave Spells (at current confidence)")
    spells = group_heatwave_spells(forecast_df)
    if spells:
        st.success("Upcoming heatwave spells detected:")
        for sp in spells:
            start, end = sp["start"].date(), sp["end"].date()
            days = sp["days"]
            st.markdown(
                f"- **{start} → {end}**  _(**{days}** day{'s' if days>1 else ''})_ — "
                f"min P={sp['min_prob']:.2f}, mean P={sp['mean_prob']:.2f}, min threshold={sp['min_threshold']:.2f}"
            )
    else:
        st.warning("No heatwave spells cross the current confidence threshold in the next 7 days.")

    # 7-day probabilities (Altair with date formatting)
    st.subheader("7-Day Forecast Probabilities")
    st.altair_chart(
        line_chart_dates(forecast_df.rename(columns={"date": "Date", "p_heatwave": "Probability"}),
                         "Date", "Probability"),
        use_container_width=True
    )

    # Forecast table with highlight
    st.subheader("Forecast Details")
    styled = forecast_df.copy()
    styled["p_heatwave"] = styled["p_heatwave"].round(3)
    styled["threshold"] = styled["threshold"].round(2)
    st.dataframe(
        styled.style.apply(
            lambda row: ["background-color: #ffe5e5" if row["predicted_heatwave"] == 1 else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
    )

    # Thresholds disclosure
    with st.expander("Thresholds in use"):
        if use_tuned and tuned:
            st.write("Using tuned per-horizon thresholds from `models/thresholds.json`:")
            st.json(tuned)
        elif use_tuned:
            st.warning("Per-horizon thresholds not found. Falling back to a fixed default of 0.50.")
        else:
            st.info("Using a fixed default threshold of 0.50 for all horizons.")

    # Recent history (Altair; exact last N days)
    st.subheader("Recent History (observed)")
    hist = test_df.copy()
    if not isinstance(hist.index, pd.DatetimeIndex):
        hist.index = pd.to_datetime(hist.index)
    end = hist.index.max()
    start = end - pd.Timedelta(days=int(show_history_days))
    hist_window = hist.loc[(hist.index >= start) & (hist.index <= end)].copy()

    candidates = [c for c in hist_window.columns if c == "tmax" or c.startswith("tmax")]
    if candidates:
        key = "tmax" if "tmax" in candidates else sorted(candidates)[0]
        plot_df = hist_window[[key]].reset_index().rename(columns={"index": "Date", key: "tmax_processed"})
        st.altair_chart(
            line_chart_dates(plot_df, "Date", "tmax_processed", title=None),
            use_container_width=True
        )
        st.caption("Note: Uses processed file (may be standardized). To plot raw °C, keep an unscaled copy during preprocessing.")
    else:
        st.write("tmax column not found in processed data to plot recent history.")


if __name__ == "__main__":
    main()
