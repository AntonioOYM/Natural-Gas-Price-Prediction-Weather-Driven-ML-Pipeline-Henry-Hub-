"""
Central project configuration.

Keeping configuration separate from logic makes experiments reproducible
and lets every module read the same parameters.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class DataConfig:
    """Data source configuration."""

    # Front-month Henry Hub natural gas futures (continuous series on Yahoo Finance)
    GAS_TICKER: str = "NG=F"

    # Historical range
    START_DATE: str = "2015-01-01"
    END_DATE: str = field(default_factory=lambda: date.today().isoformat())

    # Weather locations proxying the main gas-consuming regions.
    # Format: (name, latitude, longitude, region). Immutable tuple of tuples.
    CITIES: tuple = (
        ("New_York", 40.7128, -74.0060, "Northeast"),
        ("Chicago", 41.8781, -87.6298, "Midwest"),
        ("Houston", 29.7604, -95.3698, "South"),
        ("Los_Angeles", 34.0522, -118.2437, "West"),
        ("Denver", 39.7392, -104.9903, "Rockies"),
    )

    DATA_DIR: Path = Path("data")

    def __post_init__(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ClimateConfig:
    """Open-Meteo API configuration (free, no API key)."""

    BASE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"

    DAILY_VARIABLES: tuple = (
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "snowfall_sum",
        "precipitation_sum",
    )


@dataclass
class ResearchConfig:
    """Parameters of the evaluation protocol and the backtest."""

    # Seasonal baseline: minimum number of prior years required before an
    # anomaly is defined. Earlier observations are dropped (warm-up period).
    ANOMALY_MIN_YEARS: int = 3

    # Futures roll handling. Returns ending within +/- ROLL_TOLERANCE trading
    # days of the first trading day after estimated contract expiry are
    # excluded from training, evaluation and backtest.
    ROLL_TOLERANCE: int = 1

    # Walk-forward validation (expanding window, one calendar year per fold)
    MIN_TRAIN_YEARS: int = 3
    EMBARGO_DAYS: int = 5  # trading days removed between train end and test start
    VALID_FRACTION: float = 0.15  # tail of each training window used for early stopping

    # Backtest
    TARGET_VOL: float = 0.15  # annualised volatility target
    VOL_WINDOW: int = 20
    MAX_LEVERAGE: float = 2.0
    SLIPPAGE_TICKS: float = 1.5  # per unit of turnover
    TICK_SIZE: float = 0.001  # USD/MMBtu, NYMEX NG minimum price fluctuation
    TRADING_DAYS: int = 252


# Global instances
DATA_CFG = DataConfig()
CLIMATE_CFG = ClimateConfig()
RESEARCH_CFG = ResearchConfig()
