"""
Módulo de ingesta de datos.
"""
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from loguru import logger
from tqdm import tqdm

from config import DATA_CFG, CLIMATE_CFG


class GasPriceFetcher:
    """Descarga precios históricos de Henry Hub Natural Gas."""
    
    def __init__(self, ticker: str = DATA_CFG.GAS_TICKER):
        self.ticker = ticker
    
    def fetch(self, start: str = DATA_CFG.START_DATE, end: str = DATA_CFG.END_DATE) -> pd.DataFrame:
        """
        Descarga datos de precios desde Yahoo Finance.
        
        Returns:
            DataFrame con columnas: date, open, high, low, close, volume
        """
        logger.info(f"Descargando precios de {self.ticker} desde {start} hasta {end}")
        
        try:
            data = yf.download(
                self.ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                raise ValueError(f"No se obtuvieron datos para {self.ticker}")
            
            # Limpiar estructura (yfinance a veces devuelve MultiIndex)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data = data.reset_index()
            data = data.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            # Asegurar tipos correctos
            data["date"] = pd.to_datetime(data["date"]).dt.date
            numeric_cols = ["open", "high", "low", "close", "volume"]
            data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
            
            logger.success(f"Descargados {len(data)} registros de precios")
            return data[["date", "open", "high", "low", "close", "volume"]]
            
        except Exception as e:
            logger.error(f"Error descargando precios: {e}")
            raise


class ClimateDataFetcher:
    """Descarga datos climáticos históricos desde Open-Meteo con reintentos robustos."""
    
    def __init__(self):
        self.session = requests.Session()
        # Agregar un User-Agent ayuda a que las APIs gratuitas nos traten mejor
        self.session.headers.update({"User-Agent": "GasNaturalProject/1.0"})
    
    def fetch_city_historical(
        self,
        city_name: str,
        lat: float,
        lon: float,
        start_date: str = DATA_CFG.START_DATE,
        end_date: str = None
    ) -> pd.DataFrame:
        """Descarga datos climáticos históricos para una ciudad con reintentos exponenciales."""
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        
        logger.info(f"Descargando clima histórico para {city_name}")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(CLIMATE_CFG.DAILY_VARIABLES),
            "timezone": "auto"
        }
        
        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = self.session.get(CLIMATE_CFG.BASE_URL, params=params, timeout=30)
                
                # Si nos bloquean por velocidad, esperamos más tiempo en cada intento
                if response.status_code == 429:
                    wait_time = 10 * (attempt + 1)  # 10s, 20s, 30s, 40s
                    logger.warning(f"⚠️ Rate limit (429) para {city_name}. Esperando {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if "daily" not in data:
                    raise ValueError(f"Respuesta inesperada de Open-Meteo para {city_name}")
                
                df = pd.DataFrame(data["daily"])
                df = df.rename(columns={
                    "time": "date",
                    "temperature_2m_max": "temp_max",
                    "temperature_2m_min": "temp_min",
                    "temperature_2m_mean": "temp_mean",
                    "snowfall_sum": "snowfall_cm",
                    "precipitation_sum": "precipitation_mm"
                })
                
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df["city"] = city_name
                df["latitude"] = lat
                df["longitude"] = lon
                
                # Convertir Celsius a Fahrenheit
                for col in ["temp_max", "temp_min", "temp_mean"]:
                    df[col] = df[col] * 9/5 + 32
                
                logger.success(f"✓ Descargados {len(df)} días de clima para {city_name}")
                return df
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"❌ Error de red para {city_name} tras {max_retries} intentos: {e}")
                    raise RuntimeError(f"No se pudo descargar datos para {city_name}.")
                
                logger.warning(f"Intento {attempt + 1} fallido para {city_name}. Reintentando en 5s...")
                time.sleep(5)
        
        raise RuntimeError(f"Fallo inesperado al descargar datos para {city_name}.")

    def fetch_all_cities(self) -> pd.DataFrame:
        """Descarga clima para todas las ciudades, saltando las que fallen."""
        all_data = []
        
        for city_name, lat, lon, region in tqdm(DATA_CFG.CITIES, desc="Ciudades"):
            try:
                df = self.fetch_city_historical(city_name, lat, lon)
                df["region"] = region
                all_data.append(df)
            except Exception as e:
                logger.error(f"⚠️ Saltando {city_name} debido a error: {e}")
                continue
            
            # Pausa de 5 segundos entre ciudades para respetar la API
            time.sleep(5)
        
        if not all_data:
            raise RuntimeError("No se pudo descargar datos de ninguna ciudad. Revisa tu conexión.")
            
        return pd.concat(all_data, ignore_index=True)


class EIAStorageFetcher:
    """
    Descarga datos de almacenamiento de gas natural desde la EIA.
    Para la Fase 1 usamos un proxy público; en producción obtendrás tu key.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    
    def fetch(self, start: str = DATA_CFG.START_DATE, end: str = DATA_CFG.END_DATE) -> pd.DataFrame:
        """
        Descarga datos semanales de almacenamiento.
        
        IMPORTANTE: Para usar esto en producción, registra una API key gratuita en:
        https://www.eia.gov/opendata/requestkey.php
        """
        if not self.api_key:
            logger.warning(
                "No hay API key de EIA. Devolviendo DataFrame vacío. "
                "Regístrate en https://www.eia.gov/opendata/requestkey.php"
            )
            return pd.DataFrame(columns=["date", "value_bcf"])
        
        logger.info("Descargando datos de almacenamiento EIA")
        
        params = {
            "api_key": self.api_key,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[series][]": "NG.NW2_EPG0_SWO_R48_BCF",
            "start": start.replace("-", ""),
            "end": end.replace("-", ""),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000
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
            
            logger.success(f"Descargados {len(df)} registros de almacenamiento")
            return df[["date", "value_bcf"]]
            
        except Exception as e:
            logger.error(f"Error descargando EIA: {e}")
            return pd.DataFrame(columns=["date", "value_bcf"])