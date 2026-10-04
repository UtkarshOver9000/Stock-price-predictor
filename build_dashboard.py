"""
Rebuild the data block (`const MARKET_DATA = ...`) inside index.html from the
benchmark results, so every number on the dashboard comes from benchmark.py.

Accuracy, precision, recall, F1 and the confusion matrix cover each ticker's full
held-out test period. The charts and the strategy/benchmark returns cover the
last 252 trading days of that period.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HTML = Path("index.html")
WINDOW = 252
MODEL_KEYS = {"Random Forest": "random_forest", "Gradient Boosting": "gradient_boosting",
              "Logistic Regression": "logistic_regression"}


def _r(values, digits=2):
    return [None if not np.isfinite(v) else round(float(v), digits) for v in values]


def _pct(v):
    return round(100 * float(v), 1)


def ticker_block(meta: dict, s: dict, e: dict) -> dict:
    test, full = e["test"], e["full"]
    proba = e["proba"]
    tail = slice(len(test) - WINDOW, len(test))
    idx = test.index[tail]
    close = full["AdjClose"]
    sma20, sma50 = close.rolling(20).mean(), close.rolling(50).mean()
    std20 = close.rolling(20).std()
    window = full.loc[idx]

    preds = {k: (proba[v] >= 0.5).astype(int)[tail] for k, v in MODEL_KEYS.items()}
    rf_p = proba["random_forest"][tail]
    next_ret = (close.shift(-1) / close - 1).loc[idx].fillna(0).to_numpy()
    position = preds["Random Forest"].astype(float)
    costs = 0.001 * np.abs(np.diff(np.r_[0.0, position]))
    eq_strat = 10_000 * np.cumprod(1 + position * next_ret - costs)
    eq_bench = 10_000 * np.cumprod(1 + next_ret)

    models = {
        label: {k: _pct(s["models"][key][m]) for k, m in (("acc", "accuracy"), ("prec", "precision"),
                                                          ("rec", "recall"), ("f1", "f1"))}
        for label, key in MODEL_KEYS.items()
    }
    rf_acc = models["Random Forest"]["acc"]
    baseline = _pct(s["best_naive_baseline"])
    importances = sorted(zip(e["cols"], e["rf"].feature_importances_), key=lambda kv: -kv[1])[:10]
    return {
        "meta": meta,
        "metrics": {
            "rf_acc": rf_acc,
            "gb_acc": models["Gradient Boosting"]["acc"],
            "lr_acc": models["Logistic Regression"]["acc"],
            "baseline_maj": baseline,
            "baseline_persist": _pct(s["models"]["baseline_persistence"]["accuracy"]),
            "lift": round(rf_acc - baseline, 1),
            "strat_return": round(100 * (eq_strat[-1] / 10_000 - 1), 1),
            "bench_return": round(100 * (eq_bench[-1] / 10_000 - 1), 1),
            "total_test": int(s["test"][2]),
            "test_period": s["test"][:2],
            "cm": s["models"]["random_forest"]["confusion_matrix"],
            "models": models,
        },
        "importances": {k: round(100 * float(v), 2) for k, v in importances},
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in window["Date"]],
            "prices": _r(window["AdjClose"]),
            "pct_change": _r(100 * close.pct_change().loc[idx].fillna(0)),
            "y_true": [int(v) for v in e["y"][tail]],
            "rf_pred": [int(v) for v in preds["Random Forest"]],
            "gb_pred": [int(v) for v in preds["Gradient Boosting"]],
            "lr_pred": [int(v) for v in preds["Logistic Regression"]],
            "rf_conf": _r(100 * np.maximum(rf_p, 1 - rf_p), 1),
            "sma20": _r(sma20.loc[idx]),
            "sma50": _r(sma50.loc[idx]),
            "bb_upper": _r((sma20 + 2 * std20).loc[idx]),
            "bb_lower": _r((sma20 - 2 * std20).loc[idx]),
            "rsi": _r(window["rsi_14"], 1),
            "macd": _r(window["macd"]),
            "macd_sig": _r(window["macd_signal"]),
            "macd_hist": _r(window["macd_hist"]),
            "equity_strat": _r(eq_strat, 1),
            "equity_bench": _r(eq_bench, 1),
        },
    }


def write(summary: dict, extras: dict) -> None:
    lines = HTML.read_text(encoding="utf-8").splitlines()
    idx = next(i for i, line in enumerate(lines) if line.strip().startswith("const MARKET_DATA = "))
    old = json.loads(lines[idx].strip()[len("const MARKET_DATA = "):].rstrip(";").replace("NaN", "null"))
    data = {t: ticker_block(old[t]["meta"], summary[t], extras[t]) for t in summary}
    lines[idx] = "    const MARKET_DATA = " + json.dumps(data, separators=(",", ":")) + ";"
    HTML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"index.html: MARKET_DATA rebuilt for {len(data)} tickers")
