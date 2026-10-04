"""
Full-history benchmark: next-day direction for 12 US stocks/ETFs.

    python benchmark.py

For each ticker, on its entire daily history from Yahoo Finance:
* the first 80% of trading days are used for training (the last 15% of that
  for validation: early stopping and epoch selection), the final 20% for testing;
* models: logistic regression, random forest, histogram gradient boosting
  (validation log loss recorded every round) and an MLP (every epoch);
* baselines: majority class and persistence ("tomorrow repeats today");
* a binomial test of whether the best model's test accuracy beats the
  majority-class baseline;
* a business test: hold the stock only on days the model predicts "up",
  paying 0.1% every time the position changes, vs buy and hold.

Writes reports/benchmark.json, reports/figures/*.png and the dashboard data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from features import build_engineered_features, load_ohlcv
from labels import make_label

DATA = Path("data")
REPORTS = Path("reports")
TICKERS = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "MSFT", "GOOGL", "AMZN", "META", "AMD", "JPM", "XOM"]
TEST_FRACTION, VAL_FRACTION = 0.20, 0.15
COST = 0.001  # per change of position
SEED = 42


def _clip(p):
    return np.clip(p, 1e-7, 1 - 1e-7)


def metrics(y, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "log_loss": round(log_loss(y, _clip(proba), labels=[0, 1]), 4),
        "brier": round(brier_score_loss(y, proba), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train_boosting(xt, yt, xv, yv, rounds=400):
    params = dict(learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
                  early_stopping=False, random_state=SEED)
    full = HistGradientBoostingClassifier(max_iter=rounds, **params).fit(xt, yt)
    tr = [log_loss(yt, _clip(p[:, 1]), labels=[0, 1]) for p in full.staged_predict_proba(xt)]
    va = [log_loss(yv, _clip(p[:, 1]), labels=[0, 1]) for p in full.staged_predict_proba(xv)]
    best = int(np.argmin(va)) + 1
    return HistGradientBoostingClassifier(max_iter=best, **params).fit(xt, yt), {
        "best_round": best, "train_log_loss": [round(v, 5) for v in tr], "val_log_loss": [round(v, 5) for v in va]}


def train_mlp(xt, yt, xv, yv, epochs=60, patience=6):
    clf = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
                        random_state=SEED)
    tr, va, best, state, bad = [], [], np.inf, None, 0
    for _ in range(epochs):
        clf.partial_fit(xt, yt, classes=[0, 1])
        tr.append(log_loss(yt, _clip(clf.predict_proba(xt)[:, 1]), labels=[0, 1]))
        va.append(log_loss(yv, _clip(clf.predict_proba(xv)[:, 1]), labels=[0, 1]))
        if va[-1] < best - 1e-5:
            best, bad, state = va[-1], 0, ([w.copy() for w in clf.coefs_], [b.copy() for b in clf.intercepts_])
        else:
            bad += 1
            if bad >= patience:
                break
    clf.coefs_, clf.intercepts_ = state
    return clf, {"best_epoch": int(np.argmin(va)) + 1, "epochs_run": len(va),
                 "train_log_loss": [round(v, 5) for v in tr], "val_log_loss": [round(v, 5) for v in va]}


def strategy(test: pd.DataFrame, pred_up: np.ndarray) -> dict:
    """Long the stock for the next day when the model predicts up, otherwise cash."""
    next_ret = test["AdjClose"].shift(-1) / test["AdjClose"] - 1
    next_ret = next_ret.fillna(0).to_numpy()
    position = pred_up.astype(float)
    costs = COST * np.abs(np.diff(np.r_[0.0, position]))
    strat = position * next_ret - costs
    years = len(test) / 252

    def summary(daily):
        equity = np.cumprod(1 + daily)
        dd = equity / np.maximum.accumulate(equity) - 1
        sd = daily.std(ddof=1)
        return {
            "total_return": round(float(equity[-1] - 1), 4),
            "cagr": round(float(equity[-1] ** (1 / years) - 1), 4),
            "sharpe": round(float(daily.mean() / sd * np.sqrt(252)), 3) if sd > 0 else None,
            "max_drawdown": round(float(dd.min()), 4),
        }

    return {
        "strategy_after_costs": summary(strat),
        "buy_and_hold": summary(next_ret),
        "days_in_market": round(float(position.mean()), 4),
        "position_changes": int(np.abs(np.diff(np.r_[0.0, position])).sum()),
        "equity_strategy": np.cumprod(1 + strat),
        "equity_buy_and_hold": np.cumprod(1 + next_ret),
    }


def run_ticker(ticker: str) -> tuple[dict, dict]:
    df = load_ohlcv(str(DATA / f"{ticker}.csv"))
    feats = build_engineered_features(df)
    cols = list(feats.columns)
    full = pd.concat([df[["Date", "AdjClose", "Close"]], feats, make_label(df).rename("target")], axis=1)
    full = full.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    full["target"] = full["target"].astype(int)

    n = len(full)
    test_start = int(n * (1 - TEST_FRACTION))
    val_start = int(test_start * (1 - VAL_FRACTION))
    train, val, test = full.iloc[:val_start], full.iloc[val_start:test_start], full.iloc[test_start:]
    fit = full.iloc[:test_start]  # train + validation, for models without early stopping

    scaler = StandardScaler().fit(train[cols])
    xs = {k: scaler.transform(v[cols]) for k, v in {"train": train, "val": val, "test": test, "fit": fit}.items()}
    y = {k: v["target"].to_numpy() for k, v in {"train": train, "val": val, "test": test, "fit": fit}.items()}

    lr_scaler = StandardScaler().fit(fit[cols])
    lr = LogisticRegression(max_iter=2000).fit(lr_scaler.transform(fit[cols]), y["fit"])
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, n_jobs=-1, random_state=SEED)
    rf.fit(fit[cols], y["fit"])
    gb, gb_hist = train_boosting(train[cols], y["train"], val[cols], y["val"])
    mlp, mlp_hist = train_mlp(xs["train"], y["train"], xs["val"], y["val"])

    proba = {
        "logistic_regression": lr.predict_proba(lr_scaler.transform(test[cols]))[:, 1],
        "random_forest": rf.predict_proba(test[cols])[:, 1],
        "gradient_boosting": gb.predict_proba(test[cols])[:, 1],
        "mlp": mlp.predict_proba(xs["test"])[:, 1],
    }
    results = {name: metrics(y["test"], p) for name, p in proba.items()}
    majority = int(round(y["fit"].mean()))
    maj_pred = np.full(len(test), majority)
    results["baseline_majority"] = {"accuracy": round(accuracy_score(y["test"], maj_pred), 4), "predicts": "up" if majority else "down"}
    persist = full["target"].shift(1).iloc[test_start:].to_numpy()
    results["baseline_persistence"] = {"accuracy": round(accuracy_score(y["test"], persist), 4)}
    results["baseline_always_up"] = {"accuracy": round(float(y["test"].mean()), 4)}
    # Compare against the stronger of the two constant guesses (the conservative choice).
    naive = max(results["baseline_majority"]["accuracy"], results["baseline_always_up"]["accuracy"])

    best = max(proba, key=lambda k: results[k]["accuracy"])
    correct = int(((proba[best] >= 0.5).astype(int) == y["test"]).sum())
    p_value = binomtest(correct, len(test), naive, alternative="greater").pvalue
    trade = strategy(test, (proba["random_forest"] >= 0.5))

    out = {
        "rows": n,
        "train": [str(train["Date"].iloc[0].date()), str(train["Date"].iloc[-1].date()), len(train)],
        "validation": [str(val["Date"].iloc[0].date()), str(val["Date"].iloc[-1].date()), len(val)],
        "test": [str(test["Date"].iloc[0].date()), str(test["Date"].iloc[-1].date()), len(test)],
        "test_up_rate": round(float(y["test"].mean()), 4),
        "models": results,
        "best_model": best,
        "best_naive_baseline": naive,
        "best_minus_naive": round(results[best]["accuracy"] - naive, 4),
        "best_beats_naive_p_value": round(float(p_value), 4),
        "training": {"gradient_boosting": {"best_round": gb_hist["best_round"]},
                     "mlp": {"best_epoch": mlp_hist["best_epoch"], "epochs_run": mlp_hist["epochs_run"]}},
        "random_forest_trading": {k: v for k, v in trade.items() if not k.startswith("equity")},
    }
    extras = {"gb_hist": gb_hist, "mlp_hist": mlp_hist, "test": test, "proba": proba, "y": y["test"], "cols": cols,
              "rf": rf, "lr": lr, "lr_scaler": lr_scaler, "gb": gb, "trade": trade, "full": full}
    return out, extras


def save_figures(all_extras: dict, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = REPORTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    e = all_extras["AAPL"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(e["gb_hist"]["train_log_loss"], label="train")
    axes[0].plot(e["gb_hist"]["val_log_loss"], label="validation")
    axes[0].axvline(e["gb_hist"]["best_round"] - 1, color="grey", ls="--", label=f"kept: round {e['gb_hist']['best_round']}")
    axes[0].set(title="AAPL gradient boosting: log loss per round", xlabel="round", ylabel="log loss")
    axes[0].legend()
    ep = np.arange(1, len(e["mlp_hist"]["val_log_loss"]) + 1)
    axes[1].plot(ep, e["mlp_hist"]["train_log_loss"], marker="o", label="train")
    axes[1].plot(ep, e["mlp_hist"]["val_log_loss"], marker="o", label="validation")
    axes[1].axvline(e["mlp_hist"]["best_epoch"], color="grey", ls="--", label=f"kept: epoch {e['mlp_hist']['best_epoch']}")
    axes[1].set(title="AAPL MLP: log loss per epoch", xlabel="epoch", ylabel="log loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "aapl_training_curves.png", dpi=130)
    plt.close(fig)

    tickers = list(summary)
    best = [summary[t]["models"][summary[t]["best_model"]]["accuracy"] for t in tickers]
    base = [summary[t]["best_naive_baseline"] for t in tickers]
    xpos = np.arange(len(tickers))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(xpos - 0.2, base, 0.4, label="best constant guess (always up / majority)")
    ax.bar(xpos + 0.2, best, 0.4, label="best model")
    ax.axhline(0.5, color="grey", ls=":")
    ax.set_xticks(xpos, tickers)
    ax.set_ylim(0.4, 0.62)
    ax.set(title="Next-day direction, held-out test period: best model vs best constant guess", ylabel="accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_vs_baseline.png", dpi=130)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    summary, extras = {}, {}
    for ticker in TICKERS:
        summary[ticker], extras[ticker] = run_ticker(ticker)
        s = summary[ticker]
        print(f"{ticker}: best {s['best_model']} acc {s['models'][s['best_model']]['accuracy']} "
              f"vs naive {s['best_naive_baseline']} (p={s['best_beats_naive_p_value']})")
    meta = json.loads((DATA / "download_meta.json").read_text())
    report = {"data": meta, "label": "1 if tomorrow's adjusted close > today's, else 0",
              "split": f"chronological: train / validation ({VAL_FRACTION:.0%} of the pre-test period) / "
                       f"test (last {TEST_FRACTION:.0%})", "position_change_cost": COST,
              "total_rows": sum(s["rows"] for s in summary.values()), "tickers": summary,
              "runtime_seconds": round(time.time() - t0, 1)}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "benchmark.json").write_text(json.dumps(report, indent=2))
    save_figures(extras, summary)
    import build_dashboard

    build_dashboard.write(summary, extras)
    print("rows", report["total_rows"], "runtime", report["runtime_seconds"])


if __name__ == "__main__":
    main()
