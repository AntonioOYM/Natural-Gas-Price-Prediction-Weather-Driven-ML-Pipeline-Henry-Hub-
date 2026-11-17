"""
Configuración central del proyecto.
Separar configs del código es una buena práctica de ingeniería.
"""
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class DataConfig:
    """Configuración de fuentes de datos."""
    # Henry Hub Natural Gas Futures (NG=F en Yahoo Finance)
    GAS_TICKER: str = "NG=F"
    
    # Rango de datos históricos
    START_DATE: str = "2015-01-01"
    END_DATE: str = date.today().isoformat()
    
    # Ciudades clave para datos climáticos (representan regiones de consumo de gas)
    # Formato: (nombre, latitud, longitud, región)
    # IMPORTANTE: Usamos tupla de tuplas (inmutable) en vez de lista
    CITIES: tuple = (
        ("New_York", 40.7128, -74.0060, "Northeast"),
        ("Chicago", 41.8781, -87.6298, "Midwest"),
        ("Houston", 29.7604, -95.3698, "South"),
        ("Los_Angeles", 34.0522, -118.2437, "West"),
        ("Denver", 39.7392, -104.9903, "Rockies"),
    )
    
    # Directorio de datos
    DATA_DIR: Path = Path("data")
    
    def __post_init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)


@dataclass
class ClimateConfig:
    """Configuración para API de Open-Meteo (gratuita, sin API key)."""
    BASE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    
    # Variables climáticas que nos interesan
    # IMPORTANTE: Usamos tupla (inmutable) en vez de lista
    DAILY_VARIABLES: tuple = (
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "snowfall_sum",
        "precipitation_sum",
    )


# Instancias globales
DATA_CFG = DataConfig()
CLIMATE_CFG = ClimateConfig()