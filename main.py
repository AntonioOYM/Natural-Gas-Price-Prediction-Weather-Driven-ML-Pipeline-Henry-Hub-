"""
Full pipeline orchestrator: ingestion -> features -> walk-forward model -> backtest.

Usage: python main.py
"""

import sys

from loguru import logger

from backtest import run_backtest_stage
from config import DATA_CFG
from data_fetcher import ClimateDataFetcher, GasPriceFetcher
from feature_engineering import run_feature_engineering
from modeling import run_modeling


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{function}</cyan> - <level>{message}</level>"
        ),
        level="INFO",
    )


def main() -> None:
    setup_logging()
    logger.info("Starting full natural gas pipeline")

    logger.info("=" * 60)
    logger.info("STAGE 1: Raw data ingestion")
    df_gas = GasPriceFetcher().fetch()
    df_gas.to_parquet(DATA_CFG.DATA_DIR / "gas_prices.parquet", index=False)

    df_climate = ClimateDataFetcher().fetch_all_cities()
    df_climate.to_parquet(DATA_CFG.DATA_DIR / "climate_data.parquet", index=False)
    logger.success("Raw data saved")

    logger.info("=" * 60)
    logger.info("STAGE 2: Feature engineering")
    run_feature_engineering()

    logger.info("=" * 60)
    logger.info("STAGE 3: Walk-forward modeling (LightGBM)")
    run_modeling()

    logger.info("=" * 60)
    logger.info("STAGE 4: Backtest")
    run_backtest_stage()

    logger.info("=" * 60)
    logger.success("Pipeline finished. Results are in the 'data' directory.")


if __name__ == "__main__":
    main()
