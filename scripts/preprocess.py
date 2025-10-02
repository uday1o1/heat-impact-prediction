import argparse, os, yaml, pandas as pd
from pathlib import Path
from src.features import (
    power_json_to_df, add_calendar, label_heatwave,
    engineer_features, split_by_date, scale_and_save
)

def main(cfg):
    raw_path = cfg["data"]["raw_path"]
    interim_csv = cfg["data"]["interim_csv"]
    os.makedirs(os.path.dirname(interim_csv), exist_ok=True)

    df = power_json_to_df(raw_path)
    df = add_calendar(df)

    lab = cfg["labeling"]
    df = label_heatwave(df, abs_thresh_c=lab["tmax_abs_c"],
                        use_percentile=lab["use_percentile"],
                        win=lab["percentile_window_days"],
                        pr=lab["percentile_threshold"])

    df = engineer_features(df,
                           lags=cfg["features"]["lags_days"],
                           rolling=cfg["features"]["rolling_means_days"],
                           include_doystd=cfg["features"]["include_doystd"])

    df = df.dropna()

    # --- NEW: multi-horizon targets ---
    H = int(cfg["features"].get("target_horizons", 1))
    for h in range(1, H+1):
        df[f"heatwave_h{h}"] = df["heatwave"].shift(-h)
    df = df.dropna()

    # Feature set (exclude all target cols)
    target_cols = ["heatwave"] + [f"heatwave_h{h}" for h in range(1, H+1)]
    feature_cols = [c for c in df.columns if c not in target_cols + ["doy", "month"]]

    train, valid, test = split_by_date(df, cfg["split"]["valid_start"], cfg["split"]["test_start"])

    # Scale numeric features
    train, valid, test = scale_and_save(train, valid, test, feature_cols, cfg["data"]["scaler_path"])

    # Save parquet
    train.to_parquet(cfg["data"]["processed_train"])
    valid.to_parquet(cfg["data"]["processed_valid"])
    test.to_parquet(cfg["data"]["processed_test"])

    print("Prepared datasets:")
    print(cfg["data"]["processed_train"], cfg["data"]["processed_valid"], cfg["data"]["processed_test"])
    print("Num positives per horizon (train/valid/test):")
    for h in range(1, H+1):
        print(f"H+{h}:",
              train[f"heatwave_h{h}"].sum(),
              valid[f"heatwave_h{h}"].sum(),
              test[f"heatwave_h{h}"].sum())
    print("Feature count:", len(feature_cols))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    main(cfg)
