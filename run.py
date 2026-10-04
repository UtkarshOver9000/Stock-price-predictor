"""
Stock Price Movement Predictor - Multi-Asset Research Pipeline
================================================================
Predicts daily price direction (Up / Down) from historical OHLCV data.

Supports:
  1. Multi-Asset Real World Data: AAPL (Tech), NVDA (AI/Semis), TSLA (High Beta EV),
     SPY (S&P 500 ETF), JPM (Banking), AMZN (Cloud/Consumer).
  2. Four Labeling Variants (Strict Next-Day, 1% Dead-Zone, Overlapping 5-Day, Fair Non-Overlapping 5-Day).
  3. Visual Tools: Technical Deep-Dive (BB, RSI, MACD), Strategy Equity Curve vs. Buy & Hold,
     Confusion Matrices, ROC Curves, and Cross-Asset Comparison Matrix.
"""

from __future__ import annotations

import sys, os, argparse
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

from features import load_ohlcv, build_engineered_features, build_raw_price_features
from labels import make_label
from plots import (
    plot_predicted_vs_actual,
    plot_feature_importance,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_walk_forward_cv,
    plot_technical_deep_dive,
    plot_strategy_equity_curve,
    plot_multi_asset_matrix,
)

RANDOM_STATE = 42
TEST_FRACTION = 0.20
N_CV_FOLDS = 5

TICKER_METADATA = {
    "AAPL": "Apple Inc. (Mega-Cap Tech)",
    "NVDA": "NVIDIA Corp. (AI & Semiconductors)",
    "TSLA": "Tesla Inc. (High-Beta Growth)",
    "SPY":  "SPDR S&P 500 ETF (Broad Market)",
    "JPM":  "JPMorgan Chase (Financials / Banking)",
    "AMZN": "Amazon.com (Consumer / Cloud)",
}

VARIANTS = [
    {
        "key": "1day_strict",
        "label": "A. Next-Day Direction (No Dead Zone)",
        "horizon": 1,
        "threshold": 0.00,
        "stride": 1,
    },
    {
        "key": "1day_deadzone",
        "label": "B. Next-Day Direction (1% Dead Zone - Confident Moves)",
        "horizon": 1,
        "threshold": 0.01,
        "stride": 1,
    },
    {
        "key": "5day_strict",
        "label": "C. 5-Day Direction (Overlapping Windows - Persistence Artifact)",
        "horizon": 5,
        "threshold": 0.00,
        "stride": 1,
    },
    {
        "key": "5day_nonoverlap",
        "label": "D. 5-Day Direction (Non-Overlapping Windows - Fair Stride=5)",
        "horizon": 5,
        "threshold": 0.00,
        "stride": 5,
    },
]

ML_MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10, random_state=RANDOM_STATE),
    "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE),
}

CV_MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=RANDOM_STATE),
    "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE),
}


def assemble_dataset(data_path: str = os.path.join("data", "AAPL.csv"), horizon: int = 1, threshold: float = 0.0, stride: int = 1):
    df = load_ohlcv(data_path)
    engineered = build_engineered_features(df)
    raw = build_raw_price_features(df)
    label = make_label(df, horizon=horizon, threshold=threshold)

    full = pd.concat([df[["Date", "AdjClose"]], engineered, raw, label.rename("target")], axis=1)
    n_before = len(full)
    full = full.dropna().reset_index(drop=True)
    full["target"] = full["target"].astype(int)

    if stride > 1:
        full = full.iloc[::stride].reset_index(drop=True)

    return full, list(engineered.columns), list(raw.columns), n_before - len(full)


def time_based_split(full: pd.DataFrame, feature_cols: list):
    n = len(full)
    split_idx = int(n * (1.0 - TEST_FRACTION))
    train = full.iloc[:split_idx]
    test = full.iloc[split_idx:]
    return train[feature_cols].copy(), test[feature_cols].copy(), train["target"].copy(), test["target"].copy(), train, test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    return X_tr_s, X_te_s, scaler


def report_class_balance(y: pd.Series, name: str):
    counts = y.value_counts().sort_index()
    fracs = y.value_counts(normalize=True).sort_index()
    print(f"  {name}: Down(0) = {counts.get(0, 0)} ({fracs.get(0, 0):.1%}) | Up(1) = {counts.get(1, 0)} ({fracs.get(1, 0):.1%})")


def evaluate(y_true, y_pred, name: str, results: list, variant_key: str) -> float:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {name} ---")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Confusion Matrix [[TN, FP], [FN, TP]]:\n{cm}")

    results.append({
        "variant": variant_key,
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    })
    return acc


def walk_forward_cv(X_train_scaled, y_train: pd.Series):
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    scores = {name: [] for name in CV_MODELS}

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train_scaled), start=1):
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        for name, factory in CV_MODELS.items():
            model = factory()
            model.fit(X_tr, y_tr)
            scores[name].append(accuracy_score(y_val, model.predict(X_val)))

    print(f"\n  TimeSeriesSplit Walk-Forward CV ({N_CV_FOLDS} folds on training set):")
    for name, sc in scores.items():
        arr = [f"{s:.3f}" for s in sc]
        print(f"    {name:<22}: {arr}  mean={np.mean(sc):.3f} +/- {np.std(sc):.3f}")

    return scores


def run_variant(data_path: str, variant: dict, all_results: list, ticker: str = "AAPL") -> pd.DataFrame:
    key, label = variant["key"], variant["label"]
    horizon = variant["horizon"]
    threshold = variant["threshold"]
    stride = variant.get("stride", 1)

    print("\n" + "#" * 78)
    print(f"# [{ticker}] VARIANT {label}")
    print(f"# Horizon={horizon} trading day(s) | Threshold={threshold:.2%} | Stride={stride}")
    print("#" * 78)

    full, eng_cols, raw_cols, n_dropped = assemble_dataset(data_path, horizon, threshold, stride)
    print(f"  Usable rows: {len(full)} (dropped {n_dropped} warm-up / undefined / dead-zone rows)")
    print(f"  Engineered features ({len(eng_cols)}): {eng_cols[:4]} ... (total {len(eng_cols)})")
    print(f"  Raw-price features   ({len(raw_cols)}): {raw_cols}")

    report_class_balance(full["target"], "Full Dataset")

    Xe_tr, Xe_te, y_tr, y_te, train_df, test_df = time_based_split(full, eng_cols)
    Xr_tr, Xr_te, _,    _,    _,        _        = time_based_split(full, raw_cols)
    print(f"  Train: {train_df['Date'].iloc[0].date()} -> {train_df['Date'].iloc[-1].date()} ({len(train_df)} rows)")
    print(f"  Test : {test_df['Date'].iloc[0].date()}  -> {test_df['Date'].iloc[-1].date()}  ({len(test_df)} rows)")
    report_class_balance(y_tr, "Train Split")
    report_class_balance(y_te, "Test Split")

    Xe_tr_s, Xe_te_s, _ = scale_features(Xe_tr, Xe_te)
    Xr_tr_s, Xr_te_s, _ = scale_features(Xr_tr, Xr_te)

    cv_scores = {}
    if len(y_tr) >= 100:
        cv_scores = walk_forward_cv(Xe_tr_s, y_tr)

    results = []

    # Naive baselines
    majority = y_tr.mode()[0]
    y_maj = np.full(len(y_te), majority, dtype=int)
    evaluate(y_te, y_maj, "Baseline: Majority Class", results, key)

    y_shifted = full["target"].shift(1)
    y_persist = y_shifted.iloc[test_df.index]
    valid = y_persist.notna()
    evaluate(y_te.values[valid.values], y_persist[valid].astype(int), "Baseline: Persistence", results, key)

    # Raw model
    lr_raw = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_raw.fit(Xr_tr_s, y_tr)
    evaluate(y_te, lr_raw.predict(Xr_te_s), "Logistic Regression - RAW price features", results, key)

    # Engineered models
    trained = {}
    for name, factory in ML_MODELS.items():
        model = factory()
        if "Logistic" in name:
            model.fit(Xe_tr_s, y_tr)
            y_pred = model.predict(Xe_te_s)
        else:
            model.fit(Xe_tr, y_tr)
            y_pred = model.predict(Xe_te)
        trained[name] = (model, y_pred)
        evaluate(y_te, y_pred, f"{name} - ENGINEERED features", results, key)

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    print("\n--- VARIANT SUMMARY TABLE ---")
    print(results_df.drop(columns="variant").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Honest interpretation
    best_row = results_df.iloc[0]
    maj_acc = next(r["accuracy"] for r in results if "Majority" in r["model"])
    persist_acc = next(r["accuracy"] for r in results if "Persistence" in r["model"])
    best_baseline = max(maj_acc, persist_acc)
    lift = best_row["accuracy"] - best_baseline

    print(f"\n  Best model    : {best_row['model']}")
    print(f"  Best accuracy : {best_row['accuracy']:.4f}")
    print(f"  Best baseline : {best_baseline:.4f}")
    print(f"  Lift over base: {lift:+.4f}")

    # Generate visual artifacts
    ml_preds = {name: y_pred for name, (_, y_pred) in trained.items()}
    best_ml_name = max(ml_preds, key=lambda n: accuracy_score(y_te, ml_preds[n]))

    plot_predicted_vs_actual(test_df, y_te.values, ml_preds[best_ml_name], best_ml_name, key, f"{ticker} - {label}")
    
    rf_model, _ = trained["Random Forest"]
    plot_feature_importance(rf_model, eng_cols, key)

    models_proba = {}
    for name, (m, _) in trained.items():
        if hasattr(m, "predict_proba"):
            X_eval = Xe_te_s if "Logistic" in name else Xe_te
            models_proba[name] = m.predict_proba(X_eval)[:, 1]
    plot_roc_curves(y_te, models_proba, key)
    plot_confusion_matrices(y_te, ml_preds, key)
    plot_model_comparison(results_df, key, f"{ticker} - {label}")

    if cv_scores:
        plot_walk_forward_cv(cv_scores, key)

    # For Variant A, also generate strategy equity curve
    if key == "1day_strict":
        p_eq = plot_strategy_equity_curve(test_df, y_te.values, ml_preds[best_ml_name], ticker, best_ml_name)
        print(f"  Saved Equity Curve: {p_eq}")

    all_results.extend(results)
    return results_df


def run_cross_asset_benchmark():
    """Evaluates Next-Day direction prediction across all 6 real-world assets."""
    print("\n" + "=" * 78)
    print("RUNNING MULTI-ASSET BENCHMARK (Across 6 Diverse Real-World Sectors)")
    print("=" * 78)

    tickers = ["AAPL", "NVDA", "TSLA", "SPY", "JPM", "AMZN"]
    summary_records = []

    for t in tickers:
        p = os.path.join("data", f"{t}.csv")
        if not os.path.exists(p):
            continue
        full, eng_cols, _, _ = assemble_dataset(p, horizon=1, threshold=0.0, stride=1)
        Xe_tr, Xe_te, y_tr, y_te, _, test_df = time_based_split(full, eng_cols)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=RANDOM_STATE)
        rf.fit(Xe_tr, y_tr)
        y_pred = rf.predict(Xe_te)
        
        acc = accuracy_score(y_te, y_pred)
        # Best constant guess ("always up" or "always down") scored on the test period itself:
        # the conservative baseline a model must beat.
        maj_acc = max(y_te.mean(), 1.0 - y_te.mean())
        lift = acc - maj_acc

        summary_records.append({
            "ticker": t,
            "label": f"{t}\n({TICKER_METADATA.get(t, '').split('(')[-1].replace(')', '')})",
            "ml_accuracy": acc,
            "baseline_accuracy": maj_acc,
            "lift": lift,
        })
        print(f"  {t:<6}: ML Acc = {acc:.2%} | Baseline = {maj_acc:.2%} | Lift = {lift:+.2%}")

    df_sum = pd.DataFrame(summary_records)
    p_mat = plot_multi_asset_matrix(df_sum)
    print(f"  Saved Multi-Asset Comparison Matrix: {p_mat}")


def main():
    parser = argparse.ArgumentParser(description="Stock Price Movement Predictor")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA, SPY, JPM, AMZN)")
    parser.add_argument("--all", action="store_true", help="Run multi-asset benchmark across all downloaded tickers")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    data_path = os.path.join("data", f"{ticker}.csv")
    if not os.path.exists(data_path):
        print(f"Ticker {ticker} data not found at {data_path}. Defaulting to AAPL.csv")
        ticker = "AAPL"
        data_path = os.path.join("data", "AAPL.csv")

    os.makedirs("outputs", exist_ok=True)

    # 1. Technical deep-dive for the target ticker
    print(f"Generating Technical Indicator Deep-Dive for {ticker}...")
    df_raw = load_ohlcv(data_path)
    p_deep = plot_technical_deep_dive(df_raw, ticker)
    print(f"  Saved: {p_deep}")

    # 2. Run all 4 variants for the target ticker
    all_results = []
    for variant in VARIANTS:
        run_variant(data_path, variant, all_results, ticker)

    # 3. Save summary CSV
    combined = pd.DataFrame(all_results)
    combined.to_csv(os.path.join("outputs", "results_summary_all_variants.csv"), index=False)

    # 4. Multi-asset benchmark across all real stocks
    run_cross_asset_benchmark()

    print("\n" + "#" * 78)
    print(f"# ALL PIPELINES & VISUALIZATIONS GENERATED SUCCESSFULLY FOR {ticker}")
    print("#" * 78)


if __name__ == "__main__":
    main()
