"""
Feature engineering: turns raw weather into point-in-time predictive signals.

Every feature at date t uses only information available at or before t.
"""

import numpy as np
import pandas as pd
from loguru import logger
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from config import DATA_CFG, RESEARCH_CFG


def calculate_degree_days(df: pd.DataFrame) -> pd.DataFrame:
    """Heating (HDD) and cooling (CDD) degree days, base 65 F."""
    logger.info("Computing HDD and CDD (base 65F)...")

    df = df.copy()
    df["temp_mean"] = pd.to_numeric(df["temp_mean"], errors="coerce")

    df["hdd"] = np.maximum(0, 65 - df["temp_mean"])
    df["cdd"] = np.maximum(0, df["temp_mean"] - 65)

    return df


def calculate_anomalies(
    df: pd.DataFrame, min_years: int = RESEARCH_CFG.ANOMALY_MIN_YEARS
) -> pd.DataFrame:
    """
    Point-in-time HDD anomaly versus the seasonal norm of the same ISO week.

    The norm for (city, ISO week, year Y) is the mean over *previous* years only,
    so no information from the future of any observation enters its anomaly.
    Positive values mean colder than the historical norm for that time of year.
    Observations with fewer than `min_years` prior years of history get NaN.
    """
    logger.info(f"Computing point-in-time HDD anomalies (min {min_years} prior years)...")

    df = df.copy()
    iso = pd.to_datetime(df["date"]).dt.isocalendar()
    df["_iso_year"] = iso["year"].astype(int).to_numpy()
    df["_iso_week"] = iso["week"].astype(int).to_numpy()

    weekly = (
        df.groupby(["city", "_iso_week", "_iso_year"])["hdd"]
        .mean()
        .rename("_week_mean")
        .reset_index()
        .sort_values(["city", "_iso_week", "_iso_year"])
    )
    weekly["_baseline"] = weekly.groupby(["city", "_iso_week"])["_week_mean"].transform(
        lambda s: s.shift(1).expanding(min_periods=min_years).mean()
    )

    df = df.merge(
        weekly[["city", "_iso_week", "_iso_year", "_baseline"]],
        on=["city", "_iso_week", "_iso_year"],
        how="left",
    )
    df["hdd_anomaly"] = df["hdd"] - df["_baseline"]

    return df.drop(columns=["_iso_year", "_iso_week", "_baseline"])


def calculate_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Trailing cumulative HDD and snowfall per city (cold spells matter more than cold days)."""
    logger.info(f"Computing {window}-day rolling features...")

    df = df.copy()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    df[f"hdd_rolling_{window}d"] = df.groupby("city")["hdd"].transform(
        lambda x: x.rolling(window=window, min_periods=1).sum()
    )

    if "snowfall_cm" in df.columns:
        df[f"snowfall_rolling_{window}d"] = df.groupby("city")["snowfall_cm"].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum()
        )

    return df


def aggregate_to_daily(climate_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the city-level features into national daily signals."""
    logger.info("Aggregating weather to national daily level...")

    daily_agg = (
        climate_df.groupby("date")
        .agg(
            national_avg_hdd=("hdd", "mean"),
            national_max_hdd=("hdd", "max"),
            national_avg_cdd=("cdd", "mean"),
            national_hdd_anomaly=("hdd_anomaly", "mean"),
            national_max_hdd_7d=("hdd_rolling_7d", "max"),
            national_snowfall_7d=("snowfall_rolling_7d", "sum"),
        )
        .reset_index()
    )

    return daily_agg


def estimate_expiry_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Estimate NYMEX natural gas futures expiry dates between `start` and `end`.

    Contracts expire three business days before the first calendar day of the
    delivery month. US federal holidays are used as an approximation of the
    exchange calendar, so an estimate can be off by a day around some holidays.
    """
    cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    delivery_months = pd.date_range(
        start=pd.Timestamp(start).to_period("M").to_timestamp() - pd.offsets.MonthBegin(1),
        end=pd.Timestamp(end).to_period("M").to_timestamp() + pd.offsets.MonthBegin(2),
        freq="MS",
    )
    expiries = [first_day - 3 * cbd for first_day in delivery_months]
    return pd.DatetimeIndex(expiries)


def flag_roll_returns(dates: pd.Series, tolerance: int = RESEARCH_CFG.ROLL_TOLERANCE) -> pd.Series:
    """
    Flag trading days whose close-to-close return ends on or near a contract roll.

    On the first trading day after expiry the continuous front-month series
    switches contracts, so the return ending that day mixes two contracts and is
    not a tradeable return. Days within +/- `tolerance` trading days of that
    point are flagged too, to absorb calendar-estimation error.
    """
    raw = pd.DatetimeIndex(pd.to_datetime(dates))
    idx = raw.sort_values()
    expiries = estimate_expiry_dates(idx.min(), idx.max())

    # First trading day strictly after each estimated expiry
    positions = idx.searchsorted(expiries, side="right")
    positions = positions[(positions > 0) & (positions < len(idx))]

    flagged = np.zeros(len(idx), dtype=bool)
    for pos in positions:
        flagged[max(0, pos - tolerance) : min(len(idx), pos + tolerance + 1)] = True

    by_date = pd.Series(flagged, index=idx)
    return pd.Series(by_date.reindex(raw).to_numpy(), index=dates.index)


def roll_diagnostic(df: pd.DataFrame, max_offset: int = 4) -> pd.DataFrame:
    """
    Mean absolute daily return by trading-day offset from the estimated roll day.

    Use it to verify empirically where the continuous series actually rolls:
    a clear spike at one offset confirms the calendar assumption (offset 0 is the
    first trading day after estimated expiry).
    """
    dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    ret = df["close"].pct_change().abs().to_numpy()
    expiries = estimate_expiry_dates(dates.min(), dates.max())
    positions = dates.searchsorted(expiries, side="right")

    rows = []
    for offset in range(-max_offset, max_offset + 1):
        idx = positions + offset
        idx = idx[(idx > 0) & (idx < len(dates))]
        rows.append({"offset": offset, "mean_abs_return": np.nanmean(ret[idx]), "n": len(idx)})
    rows.append({"offset": "all days", "mean_abs_return": np.nanmean(ret), "n": len(ret)})
    return pd.DataFrame(rows)


def merge_and_clean(gas_df: pd.DataFrame, climate_daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join prices with weather features and add market-derived features.

    - Prices after the last available weather date are dropped, so no row relies
      on forward-filled (stale) weather.
    - Rows in the anomaly warm-up period (no seasonal baseline yet) are dropped.
    """
    logger.info("Merging datasets and cleaning...")

    gas_df = gas_df.copy()
    gas_df["date"] = pd.to_datetime(gas_df["date"]).dt.date

    climate_daily_df = climate_daily_df.copy()
    climate_daily_df["date"] = pd.to_datetime(climate_daily_df["date"]).dt.date

    last_weather_date = climate_daily_df["date"].max()
    n_before = len(gas_df)
    gas_df = gas_df[gas_df["date"] <= last_weather_date]
    if len(gas_df) < n_before:
        logger.info(f"Dropped {n_before - len(gas_df)} price rows after last weather date")

    merged = pd.merge(gas_df, climate_daily_df, on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.dropna(subset=["close"])

    # Isolated gaps in weather (e.g. a missing day) are filled with the last known value
    price_cols = ["date", "open", "high", "low", "close", "volume"]
    weather_cols = [c for c in merged.columns if c not in price_cols]
    merged[weather_cols] = merged[weather_cols].ffill()

    # Volume relative to its trailing 20-day mean (raw volume is non-stationary)
    merged["volume_ratio_20d"] = merged["volume"] / merged["volume"].rolling(
        20, min_periods=10
    ).mean().replace(0, np.nan)

    # Returns ending on a contract roll are not tradeable returns
    merged["roll_return_flag"] = flag_roll_returns(merged["date"]).to_numpy()

    n_before = len(merged)
    merged = merged.dropna(subset=["national_hdd_anomaly"]).reset_index(drop=True)
    logger.info(f"Dropped {n_before - len(merged)} warm-up rows without a seasonal baseline")

    logger.success(f"Final dataset: {len(merged)} rows, {len(merged.columns)} columns.")
    return merged


def run_feature_engineering() -> pd.DataFrame:
    """Feature-engineering pipeline orchestrator."""
    logger.info("Starting feature engineering")

    gas_df = pd.read_parquet(DATA_CFG.DATA_DIR / "gas_prices.parquet")
    climate_df = pd.read_parquet(DATA_CFG.DATA_DIR / "climate_data.parquet")

    climate_df = calculate_degree_days(climate_df)
    climate_df = calculate_anomalies(climate_df)
    climate_df = calculate_rolling_features(climate_df, window=7)

    climate_daily_df = aggregate_to_daily(climate_df)
    final_df = merge_and_clean(gas_df, climate_daily_df)

    output_path = DATA_CFG.DATA_DIR / "final_dataset.parquet"
    final_df.to_parquet(output_path, index=False)
    logger.success(f"Final dataset saved to: {output_path}")

    return final_df
