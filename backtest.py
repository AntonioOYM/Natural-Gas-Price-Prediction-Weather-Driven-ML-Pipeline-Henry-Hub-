"""
Backtest: turns out-of-sample predictions into a volatility-targeted long/short
strategy with explicit transaction costs.

Timing convention
-----------------
A prediction made at the close of day t sets the position held from t to t+1 and
earns the realised return y_true(t) = close(t+1) / close(t) - 1. Position sizing
uses volatility measured with returns up to and including day t only.

Costs are charged as slippage on turnover: `SLIPPAGE_TICKS` ticks of
`TICK_SIZE` USD/MMBtu per unit of position change, expressed relative to price.
Commissions, financing and margin are not modelled.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from config import DATA_CFG, RESEARCH_CFG


@dataclass
class BacktestResult:
    """Daily strategy series and summary statistics."""

    daily: pd.DataFrame
    summary: dict


def trailing_volatility(prices: pd.DataFrame, window: int = RESEARCH_CFG.VOL_WINDOW) -> pd.Series:
    """
    Annualised trailing volatility of daily returns, indexed by date.

    Returns that end on a futures roll are excluded because they are artefacts
    of the continuous series, not market moves.
    """
    prices = prices.sort_values("date")
    returns = prices["close"].pct_change()
    if "roll_return_flag" in prices.columns:
        returns = returns.mask(prices["roll_return_flag"].astype(bool))
    vol = returns.rolling(window, min_periods=max(5, window // 2)).std()
    vol = vol * np.sqrt(RESEARCH_CFG.TRADING_DAYS)
    return pd.Series(vol.to_numpy(), index=pd.to_datetime(prices["date"]).to_numpy())


def _sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=1)
    if len(returns) < 2 or not std > 0:
        return float("nan")
    return float(returns.mean() / std * np.sqrt(RESEARCH_CFG.TRADING_DAYS))


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def run_backtest(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float = RESEARCH_CFG.TARGET_VOL,
    max_leverage: float = RESEARCH_CFG.MAX_LEVERAGE,
    slippage_ticks: float = RESEARCH_CFG.SLIPPAGE_TICKS,
    tick_size: float = RESEARCH_CFG.TICK_SIZE,
) -> BacktestResult:
    """
    Backtest the sign of the out-of-sample prediction.

    Args:
        predictions: columns date, fold, test_year, close, y_true, y_pred.
        prices: full price history (date, close[, roll_return_flag]) used only
            to measure trailing volatility.
    """
    df = predictions.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])

    vol = trailing_volatility(prices)
    df["vol_ann"] = vol.reindex(df["date"]).to_numpy()

    leverage = (target_vol / df["vol_ann"]).clip(upper=max_leverage).fillna(0.0)
    df["signal"] = np.sign(df["y_pred"])
    df["position"] = df["signal"] * leverage

    df["gross_return"] = df["position"] * df["y_true"]
    df["turnover"] = df["position"].diff().abs().fillna(df["position"].abs())
    df["cost"] = df["turnover"] * slippage_ticks * tick_size / df["close"]
    df["net_return"] = df["gross_return"] - df["cost"]

    # Benchmark: always long, same volatility targeting and no turnover costs
    df["benchmark_return"] = leverage * df["y_true"]

    df["equity_net"] = (1 + df["net_return"]).cumprod()
    df["equity_gross"] = (1 + df["gross_return"]).cumprod()
    df["equity_benchmark"] = (1 + df["benchmark_return"]).cumprod()

    ann = RESEARCH_CFG.TRADING_DAYS
    n = len(df)
    years = n / ann
    sharpe_net = _sharpe(df["net_return"])
    fold_sharpes = df.groupby("fold")["net_return"].apply(_sharpe)

    summary = {
        "n_days": n,
        "target_vol": target_vol,
        "slippage_ticks": slippage_ticks,
        "sharpe_gross": _sharpe(df["gross_return"]),
        "sharpe_net": sharpe_net,
        "sharpe_net_approx_stderr": float(np.sqrt((1 + 0.5 * sharpe_net**2) / years))
        if years > 0 and sharpe_net == sharpe_net
        else float("nan"),
        "sharpe_benchmark_long": _sharpe(df["benchmark_return"]),
        "ann_return_net": float(df["net_return"].mean() * ann),
        "ann_vol_net": float(df["net_return"].std(ddof=1) * np.sqrt(ann)),
        "max_drawdown_net": _max_drawdown(df["equity_net"]),
        "hit_rate_active_days": float((df.loc[df["position"] != 0, "net_return"] > 0).mean()),
        "avg_daily_turnover": float(df["turnover"].mean()),
        "total_cost_return": float(df["cost"].sum()),
        "fold_sharpe_net_mean": float(fold_sharpes.mean()),
        "fold_sharpe_net_std": float(fold_sharpes.std(ddof=1)),
        "fold_sharpe_net": {int(k): float(v) for k, v in fold_sharpes.items()},
    }
    return BacktestResult(daily=df, summary=summary)


def run_backtest_stage() -> BacktestResult:
    """Load predictions and prices, run the backtest and persist results."""
    logger.info("Starting backtest stage")
    out = DATA_CFG.DATA_DIR

    predictions = pd.read_parquet(out / "oos_predictions.parquet")
    prices = pd.read_parquet(out / "final_dataset.parquet")
    result = run_backtest(predictions, prices)

    result.daily.to_parquet(out / "backtest_daily.parquet", index=False)
    (out / "backtest_metrics.json").write_text(json.dumps(result.summary, indent=2))

    s = result.summary
    logger.success("Backtest results (out-of-sample, net of slippage)")
    logger.info(f"  Sharpe gross / net   : {s['sharpe_gross']:.2f} / {s['sharpe_net']:.2f}")
    logger.info(f"  Sharpe long benchmark: {s['sharpe_benchmark_long']:.2f}")
    logger.info(f"  max drawdown         : {s['max_drawdown_net']:.1%}")
    logger.info(
        f"  fold Sharpe mean/std : {s['fold_sharpe_net_mean']:.2f} / {s['fold_sharpe_net_std']:.2f}"
    )
    return result
