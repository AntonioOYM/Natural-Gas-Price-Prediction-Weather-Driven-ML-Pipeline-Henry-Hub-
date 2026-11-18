"""
Modeling: walk-forward evaluation of a LightGBM model on next-day returns.

Protocol
--------
* Expanding-window walk-forward: for each calendar year Y after a minimum
  training history, train on all data before Y (minus an embargo) and predict Y.
* Early stopping uses a validation block carved from the END of each training
  window, never the test year.
* Out-of-sample predictions from all folds are pooled for the final metrics, and
  per-fold metrics give a distribution (mean and standard deviation).
"""

import json
from dataclasses import dataclass

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from scipy import stats  # noqa: E402

from config import DATA_CFG, RESEARCH_CFG  # noqa: E402

FEATURE_COLS = [
    "national_avg_hdd",
    "national_max_hdd",
    "national_avg_cdd",
    "national_hdd_anomaly",
    "national_max_hdd_7d",
    "national_snowfall_7d",
    "volume_ratio_20d",
]

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "min_child_samples": 50,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}


@dataclass
class WalkForwardResult:
    """Container for the outputs of a walk-forward run."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: dict
    importance: pd.Series


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Next-day return target, aligned so that features at t predict the return t -> t+1.

    Rows whose target return crosses a futures roll are removed because the
    continuous series mixes two contracts there. The last row (no successor)
    is removed too.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])

    df["target_next_day_return"] = df["close"].pct_change().shift(-1)

    if "roll_return_flag" in df.columns:
        target_is_roll = df["roll_return_flag"].shift(-1, fill_value=False).astype(bool)
    else:
        target_is_roll = pd.Series(False, index=df.index)
    df["target_roll_flag"] = target_is_roll

    n_roll = int(target_is_roll.sum())
    df = df[~target_is_roll].dropna(subset=["target_next_day_return"]).reset_index(drop=True)
    logger.info(f"Target built: {len(df)} rows ({n_roll} roll-affected returns excluded)")

    df["target_direction"] = (df["target_next_day_return"] > 0).astype(int)
    return df


def walk_forward_splits(
    dates: pd.Series,
    min_train_years: int = RESEARCH_CFG.MIN_TRAIN_YEARS,
    embargo_days: int = RESEARCH_CFG.EMBARGO_DAYS,
    min_train_rows: int = 200,
) -> list[dict]:
    """
    Expanding-window splits with one calendar year per test fold.

    Rows must be sorted by date. Training rows end `embargo_days` rows before the
    first test row, so no training target overlaps the test period.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted in ascending order")

    years = dates.dt.year
    first_test_year = years.min() + min_train_years

    splits = []
    for year in sorted(years.unique()):
        if year < first_test_year:
            continue
        test_idx = np.flatnonzero((years == year).to_numpy())
        train_end = test_idx[0] - embargo_days
        if train_end < min_train_rows:
            continue
        splits.append(
            {
                "fold": len(splits) + 1,
                "test_year": int(year),
                "train_idx": np.arange(0, train_end),
                "test_idx": test_idx,
            }
        )
    return splits


def _fit_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    valid_fraction: float = RESEARCH_CFG.VALID_FRACTION,
) -> lgb.Booster:
    """Train on the head of the window, early-stop on its chronological tail."""
    n_valid = max(1, int(len(X_train) * valid_fraction))
    split = len(X_train) - n_valid

    train_set = lgb.Dataset(X_train.iloc[:split], label=y_train.iloc[:split])
    valid_set = lgb.Dataset(X_train.iloc[split:], label=y_train.iloc[split:], reference=train_set)

    return lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics for one set of out-of-sample predictions."""
    up_rate = float((y_true > 0).mean())
    hit = (y_pred > 0) == (y_true > 0)
    ic = stats.spearmanr(y_pred, y_true).statistic if np.std(y_pred) > 0 else np.nan
    return {
        "n": int(len(y_true)),
        "directional_accuracy": float(hit.mean()),
        "up_day_rate": up_rate,
        "naive_accuracy": max(up_rate, 1 - up_rate),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae_zero_forecast": float(np.mean(np.abs(y_true))),
        "information_coefficient": float(ic) if ic == ic else np.nan,
    }


def _summarise(predictions: pd.DataFrame, fold_metrics: pd.DataFrame) -> dict:
    """Pooled out-of-sample metrics plus significance tests."""
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    pooled = _fold_metrics(y_true, y_pred)

    n = pooled["n"]
    hits = int(round(pooled["directional_accuracy"] * n))
    p_vs_coin = stats.binomtest(hits, n, 0.5, alternative="greater").pvalue
    p_vs_naive = stats.binomtest(hits, n, pooled["naive_accuracy"], alternative="greater").pvalue

    return {
        "n_folds": int(len(fold_metrics)),
        "n_oos_observations": n,
        "oos_period": [
            str(predictions["date"].min().date()),
            str(predictions["date"].max().date()),
        ],
        "pooled_directional_accuracy": pooled["directional_accuracy"],
        "up_day_base_rate": pooled["up_day_rate"],
        "naive_majority_accuracy": pooled["naive_accuracy"],
        "pvalue_vs_coin_flip": float(p_vs_coin),
        "pvalue_vs_naive_majority": float(p_vs_naive),
        "pooled_mae": pooled["mae"],
        "pooled_mae_zero_forecast": pooled["mae_zero_forecast"],
        "pooled_rmse": pooled["rmse"],
        "pooled_information_coefficient": pooled["information_coefficient"],
        "fold_accuracy_mean": float(fold_metrics["directional_accuracy"].mean()),
        "fold_accuracy_std": float(fold_metrics["directional_accuracy"].std(ddof=1)),
        "fold_ic_mean": float(fold_metrics["information_coefficient"].mean()),
        "fold_ic_std": float(fold_metrics["information_coefficient"].std(ddof=1)),
    }


def run_walk_forward(df: pd.DataFrame) -> WalkForwardResult:
    """Run the expanding-window walk-forward evaluation on a target-augmented dataset."""
    splits = walk_forward_splits(df["date"])
    if not splits:
        raise ValueError("Not enough history for walk-forward evaluation")

    X = df[FEATURE_COLS]
    y = df["target_next_day_return"]

    pred_frames, metric_rows, importances = [], [], []
    for split in splits:
        tr, te = split["train_idx"], split["test_idx"]
        model = _fit_fold(X.iloc[tr], y.iloc[tr])
        y_pred = model.predict(X.iloc[te], num_iteration=model.best_iteration)

        frame = pd.DataFrame(
            {
                "date": df["date"].iloc[te].to_numpy(),
                "fold": split["fold"],
                "test_year": split["test_year"],
                "close": df["close"].iloc[te].to_numpy(),
                "y_true": y.iloc[te].to_numpy(),
                "y_pred": y_pred,
            }
        )
        pred_frames.append(frame)

        metrics = _fold_metrics(frame["y_true"].to_numpy(), y_pred)
        metrics.update(
            fold=split["fold"],
            test_year=split["test_year"],
            n_train=int(len(tr)),
            best_iteration=int(model.best_iteration or 0),
        )
        metric_rows.append(metrics)
        importances.append(
            pd.Series(model.feature_importance(importance_type="gain"), index=FEATURE_COLS)
        )
        logger.info(
            f"Fold {split['fold']} (test {split['test_year']}): "
            f"accuracy={metrics['directional_accuracy']:.2%}, n={metrics['n']}"
        )

    predictions = pd.concat(pred_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    importance = importance / importance.sum()

    return WalkForwardResult(
        predictions=predictions,
        fold_metrics=fold_metrics,
        summary=_summarise(predictions, fold_metrics),
        importance=importance,
    )


def _plot_importance(importance: pd.Series, path) -> None:
    ordered = importance.sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered.index, ordered.values, color="#2E5C8A")
    ax.set_xlabel("Share of total gain (mean across walk-forward folds)")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_modeling() -> WalkForwardResult:
    """Modeling-stage orchestrator: load, build target, evaluate, persist."""
    logger.info("Starting modeling stage (walk-forward)")

    df = pd.read_parquet(DATA_CFG.DATA_DIR / "final_dataset.parquet")
    df = create_target_variable(df)
    result = run_walk_forward(df)

    out = DATA_CFG.DATA_DIR
    result.predictions.to_parquet(out / "oos_predictions.parquet", index=False)
    result.fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    result.importance.rename("gain_share").to_csv(out / "feature_importance.csv")
    (out / "metrics.json").write_text(json.dumps(result.summary, indent=2))
    _plot_importance(result.importance, out / "feature_importance.png")

    s = result.summary
    logger.success("Walk-forward results (pooled out-of-sample)")
    logger.info(f"  folds / observations : {s['n_folds']} / {s['n_oos_observations']}")
    logger.info(f"  directional accuracy : {s['pooled_directional_accuracy']:.2%}")
    logger.info(f"  naive majority       : {s['naive_majority_accuracy']:.2%}")
    logger.info(
        f"  p-value vs coin flip : {s['pvalue_vs_coin_flip']:.3f} | "
        f"vs naive: {s['pvalue_vs_naive_majority']:.3f}"
    )
    logger.info(f"  information coeff.   : {s['pooled_information_coefficient']:.4f}")
    logger.info(
        f"  MAE model / zero fc  : {s['pooled_mae']:.5f} / {s['pooled_mae_zero_forecast']:.5f}"
    )
    return result
