"""
Data ingestion: Henry Hub futures prices (Yahoo Finance) and daily weather (Open-Meteo).
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
from loguru import logger
from tqdm import tqdm

from config import CLIMATE_CFG, DATA_CFG


class GasPriceFetcher:
    """Downloads historical front-month Henry Hub natural gas futures prices."""

    def __init__(self, ticker: str = DATA_CFG.GAS_TICKER) -> None:
        self.ticker = ticker

    def fetch(self, start: str = DATA_CFG.START_DATE, end: str = DATA_CFG.END_DATE) -> pd.DataFrame:
        """
        Download daily prices from Yahoo Finance.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume.
        """
        logger.info(f"Downloading {self.ticker} prices from {start} to {end}")

        try:
            data = yf.download(
                self.ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                raise ValueError(f"No data returned for {self.ticker}")

            # yfinance sometimes returns a MultiIndex on columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.reset_index()
            data = data.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            data["date"] = pd.to_datetime(data["date"]).dt.date
            numeric_cols = ["open", "high", "low", "close", "volume"]
            data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")

            logger.success(f"Downloaded {len(data)} price records")
            return data[["date", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.error(f"Error downloading prices: {e}")
            raise


class ClimateDataFetcher:
    """Downloads historical weather from Open-Meteo with exponential retry."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NatGasWeatherResearch/1.0"})

    def fetch_city_historical(
        self,
        city_name: str,
        lat: float,
        lon: float,
        start_date: str = DATA_CFG.START_DATE,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Download historical daily weather for one location."""
        if end_date is None:
            # The archive API lags real time by a couple of days
            end_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

        logger.info(f"Downloading historical weather for {city_name}")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(CLIMATE_CFG.DAILY_VARIABLES),
            "timezone": "auto",
        }

        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = self.session.get(CLIMATE_CFG.BASE_URL, params=params, timeout=30)

                if response.status_code == 429:
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"Rate limit (429) for {city_name}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                if "daily" not in data:
                    raise ValueError(f"Unexpected Open-Meteo response for {city_name}")

                df = pd.DataFrame(data["daily"])
                df = df.rename(
                    columns={
                        "time": "date",
                        "temperature_2m_max": "temp_max",
                        "temperature_2m_min": "temp_min",
                        "temperature_2m_mean": "temp_mean",
                        "snowfall_sum": "snowfall_cm",
                        "precipitation_sum": "precipitation_mm",
                    }
                )

                df["date"] = pd.to_datetime(df["date"]).dt.date
                df["city"] = city_name
                df["latitude"] = lat
                df["longitude"] = lon

                # Celsius -> Fahrenheit
                for col in ["temp_max", "temp_min", "temp_mean"]:
                    df[col] = df[col] * 9 / 5 + 32

                logger.success(f"Downloaded {len(df)} days of weather for {city_name}")
                return df

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Network error for {city_name} after {max_retries} attempts: {e}")
                    raise RuntimeError(f"Could not download data for {city_name}.") from e

                logger.warning(f"Attempt {attempt + 1} failed for {city_name}. Retrying in 5s...")
                time.sleep(5)

        raise RuntimeError(f"Unexpected failure downloading data for {city_name}.")

    def fetch_all_cities(self) -> pd.DataFrame:
        """Download weather for all configured cities, skipping those that fail."""
        all_data = []

        for city_name, lat, lon, region in tqdm(DATA_CFG.CITIES, desc="Cities"):
            try:
                df = self.fetch_city_historical(city_name, lat, lon)
                df["region"] = region
                all_data.append(df)
            except Exception as e:
                logger.error(f"Skipping {city_name} due to error: {e}")
                continue

            # Pause between cities to respect the free API
            time.sleep(5)

        if not all_data:
            raise RuntimeError("Could not download data for any city. Check your connection.")

        return pd.concat(all_data, ignore_index=True)


class EIAStorageFetcher:
    """
    EXPERIMENTAL and not wired into the pipeline.

    Sketch of a weekly EIA storage client. The series identifier and facet names
    have not been validated against the live API, and publication-lag handling
    (reports are released on Thursdays for the week ending the previous Friday)
    is not implemented. See the project roadmap.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"

    def fetch(self, start: str = DATA_CFG.START_DATE, end: str = DATA_CFG.END_DATE) -> pd.DataFrame:
        """Download weekly storage data. Requires a free key from eia.gov/opendata."""
        if not self.api_key:
            logger.warning("No EIA API key provided. Returning an empty DataFrame.")
            return pd.DataFrame(columns=["date", "value_bcf"])

        logger.info("Downloading EIA storage data")

        params = {
            "api_key": self.api_key,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[series][]": "NW2_EPG0_SWO_R48_BCF",
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            records = data.get("response", {}).get("data", [])
            df = pd.DataFrame(records)

            if df.empty:
                return pd.DataFrame(columns=["date", "value_bcf"])

            df = df.rename(columns={"period": "date", "value": "value_bcf"})
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["value_bcf"] = pd.to_numeric(df["value_bcf"], errors="coerce")

            logger.success(f"Downloaded {len(df)} storage records")
            return df[["date", "value_bcf"]]

        except Exception as e:
            logger.error(f"Error downloading EIA data: {e}")
            return pd.DataFrame(columns=["date", "value_bcf"])
