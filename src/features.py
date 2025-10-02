import json, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def power_json_to_df(raw_path: str) -> pd.DataFrame:
    with open(raw_path) as f:
        j = json.load(f)
    p = j["properties"]["parameter"]
    # Align by dates
    dates = pd.to_datetime(list(next(iter(p.values())).keys()))
    df = pd.DataFrame(index=dates)
    rename = {
        "T2M_MAX": "tmax",
        "T2M_MIN": "tmin",
        "RH2M": "rh",
        "WS2M": "ws",
        "ALLSKY_SFC_SW_DWN": "srad"
    }
    for k, v in p.items():
        if k in rename:
            s = pd.Series(v, dtype="float64")
            s.index = dates
            df[rename[k]] = s.values
    df = df.sort_index().asfreq("D")
    return df

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["doy"] = df.index.dayofyear
    df["month"] = df.index.month
    return df

def label_heatwave(df, abs_thresh_c=40.0, use_percentile=True, win=31, pr=0.95):
    t = df["tmax"]
    label = (t >= abs_thresh_c)
    if use_percentile:
        # Rolling seasonal 31-day window percentile by day-of-year
        t_s = t.copy()
        # Fill with rolling median for robustness
        t_roll = t_s.rolling(win, center=True, min_periods=10).quantile(pr)
        # Backfill gaps at edges
        t_roll = t_roll.bfill().ffill()
        label = label | (t >= t_roll)
    df["heatwave"] = label.astype(int)
    return df

def engineer_features(df, lags=[1,2,3,7], rolling=[3,7], include_doystd=True):
    # simple safety fills
    df = df.copy()
    for col in ["tmax","tmin","rh","ws","srad"]:
        if col in df:
            df[col] = df[col].interpolate(limit_direction="both")

    for k in lags:
        for col in ["tmax","rh","ws","srad"]:
            if col in df:
                df[f"{col}_lag{k}"] = df[col].shift(k)

    for w in rolling:
        for col in ["tmax","rh","ws","srad"]:
            if col in df:
                df[f"{col}_ma{w}"] = df[col].rolling(w).mean()

    if include_doystd:
        # anomaly relative to 31d rolling std
        df["tmax_std31"] = df["tmax"].rolling(31, center=True, min_periods=10).std()
        df["tmax_anom_std31"] = (df["tmax"] - df["tmax"].rolling(31, center=True, min_periods=10).mean()) / (df["tmax_std31"] + 1e-6)

    return df

def split_by_date(df, valid_start, test_start):
    train = df[df.index < valid_start]
    valid = df[(df.index >= valid_start) & (df.index < test_start)]
    test  = df[df.index >= test_start]
    return train, valid, test

def scale_and_save(train_df, valid_df, test_df, feature_cols, scaler_path):
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].values)
    valid_df[feature_cols] = scaler.transform(valid_df[feature_cols].values)
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols].values)
    Path(os.path.dirname(scaler_path)).mkdir(parents=True, exist_ok=True)
    import joblib; joblib.dump(scaler, scaler_path)
    return train_df, valid_df, test_df
