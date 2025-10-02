import argparse
import pandas as pd

def badge(x, good=0.90, ok=0.80):
    # Emoji badge by score (used for ROC-AUC and PR-AUC)
    if pd.isna(x): return " "
    return "🟢" if x >= good else ("🟡" if x >= ok else "🟠")

def main(args):
    df = pd.read_csv(args.metrics_csv)

    # Round & prepare
    df["prevalence"] = (df["prevalence"] * 100).map(lambda v: f"{v:,.1f}%")
    for col in ["roc_auc", "pr_auc", "f1_at_threshold"]:
        df[col] = df[col].map(lambda v: f"{v:.3f}")
    df["threshold"] = df["threshold"].map(lambda v: f"{v:.2f}")

    # Badges (qualitative hint)
    roc_badge = df["roc_auc"].astype(float).map(badge)
    pr_badge  = df["pr_auc"].astype(float).map(badge)

    # Build aligned Markdown table
    header = [
        "| Horizon | Prev | ROC-AUC |  | PR-AUC |  | F1@thr | thr |",
        "|:--:|--:|--:|:--:|--:|:--:|--:|--:|",
    ]
    rows = []
    for i, r in df.iterrows():
        rows.append(
            f"| {r['horizon']} "
            f"| {r['prevalence']} "
            f"| {r['roc_auc']} | {roc_badge.iloc[i]} "
            f"| {r['pr_auc']} | {pr_badge.iloc[i]} "
            f"| {r['f1_at_threshold']} "
            f"| {r['threshold']} |"
        )

    out_md = "\n".join(header + rows)

    with open(args.outmd, "w") as f:
        f.write(out_md)

    print("Wrote:", args.outmd)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", default="reports/metrics/test_metrics.csv")
    ap.add_argument("--outmd", default="reports/metrics/METRICS.md")
    args = ap.parse_args()
    main(args)
