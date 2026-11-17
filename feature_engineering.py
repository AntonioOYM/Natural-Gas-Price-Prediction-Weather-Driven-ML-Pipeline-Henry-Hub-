"""
Módulo de Feature Engineering.
Transforma datos climáticos crudos en señales predictivas (HDD, CDD, Anomalías, Rolling Windows).
"""
import pandas as pd
import numpy as np
from loguru import logger
from typing import Tuple

from config import DATA_CFG


def calculate_degree_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Heating Degree Days (HDD) y Cooling Degree Days (CDD).
    Base de confort: 65°F.
    """
    logger.info("Calculando HDD y CDD (Base 65°F)...")
    
    # Asegurarnos de que temp_mean sea numérico
    df = df.copy()
    df["temp_mean"] = pd.to_numeric(df["temp_mean"], errors="coerce")
    
    # HDD: Si temp_mean < 65, la diferencia es positiva. Si no, 0.
    df["hdd"] = np.maximum(0, 65 - df["temp_mean"])
    
    # CDD: Si temp_mean > 65, la diferencia es positiva. Si no, 0.
    df["cdd"] = np.maximum(0, df["temp_mean"] - 65)
    
    return df


def calculate_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la anomalía del HDD comparado con el promedio histórico de esa semana del año.
    Una anomalía positiva significa "más frío de lo normal para esta época".
    """
    logger.info("Calculando anomalías históricas de HDD...")
    
    df = df.copy()
    
    # Extraer la semana del año (1-53) para agrupar
    df["week_of_year"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)
    
    # Calcular el promedio histórico de HDD por ciudad y por semana del año
    # (excluyendo el año actual para evitar data leakage, aunque para este demo usamos todo)
    historical_avg = df.groupby(["city", "week_of_year"])["hdd"].transform("mean")
    
    df["hdd_anomaly"] = df["hdd"] - historical_avg
    
    # Limpiar columna temporal
    df = df.drop(columns=["week_of_year"])
    
    return df


def calculate_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Calcula características de ventana móvil (ej: HDD acumulado en los últimos 7 días).
    Una ola de frío de 7 días impacta los inventarios de gas mucho más que 1 día frío.
    """
    logger.info(f"Calculando características de ventana móvil ({window} días)...")
    
    df = df.copy()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    
    # HDD acumulado a 7 días por ciudad
    df[f"hdd_rolling_{window}d"] = df.groupby("city")["hdd"].transform(
        lambda x: x.rolling(window=window, min_periods=1).sum()
    )
    
    # Snowfall acumulado a 7 días por ciudad
    if "snowfall_cm" in df.columns:
        df[f"snowfall_rolling_{window}d"] = df.groupby("city")["snowfall_cm"].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum()
        )
    
    return df


def aggregate_to_daily(climate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los datos de las 5 ciudades a métricas diarias nacionales/regionales.
    Henry Hub reacciona al estrés térmico agregado de las principales zonas de consumo.
    """
    logger.info("Agregando datos climáticos a nivel diario nacional...")
    
    # Agrupamos por fecha y calculamos estadísticas clave entre las ciudades
    daily_agg = climate_df.groupby("date").agg(
        avg_hdd=("hdd", "mean"),          # Promedio de estrés de calefacción en las 5 regiones
        max_hdd=("hdd", "max"),           # La región con más frío (suele impulsar el precio)
        avg_cdd=("cdd", "mean"),          # Promedio de estrés de enfriamiento
        avg_hdd_anomaly=("hdd_anomaly", "mean"), # Anomalía promedio
        max_hdd_rolling_7d=("hdd_rolling_7d", "max"), # Máximo estrés acumulado semanal
        total_snowfall_7d=("snowfall_rolling_7d", "sum") # Nieve total acumulada en las regiones
    ).reset_index()
    
    # Renombrar para que sea más limpio
    daily_agg = daily_agg.rename(columns={
        "avg_hdd": "national_avg_hdd",
        "max_hdd": "national_max_hdd",
        "avg_cdd": "national_avg_cdd",
        "avg_hdd_anomaly": "national_hdd_anomaly",
        "max_hdd_rolling_7d": "national_max_hdd_7d",
        "total_snowfall_7d": "national_snowfall_7d"
    })
    
    return daily_agg


def merge_and_clean(gas_df: pd.DataFrame, climate_daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une los precios del gas con las features climáticas y limpia valores faltantes.
    """
    logger.info("Uniendo datasets y limpiando valores faltantes...")
    
    # Asegurar que las fechas sean del mismo tipo
    gas_df = gas_df.copy()
    gas_df["date"] = pd.to_datetime(gas_df["date"]).dt.date
    
    climate_daily_df = climate_daily_df.copy()
    climate_daily_df["date"] = pd.to_datetime(climate_daily_df["date"]).dt.date
    
    # Merge a la izquierda (queremos todos los días de trading de gas)
    merged_df = pd.merge(gas_df, climate_daily_df, on="date", how="left")
    
    # Ordenar por fecha
    merged_df = merged_df.sort_values("date").reset_index(drop=True)
    
    # Forward fill para valores faltantes de clima (ej: fines de semana sin datos de alguna API)
    # Es válido porque el clima no cambia drásticamente de un día al otro en términos de tendencia
    climate_cols = [col for col in merged_df.columns if col not in ["date", "open", "high", "low", "close", "volume"]]
    merged_df[climate_cols] = merged_df[climate_cols].ffill()
    
    # Drop rows where 'close' price is NaN (días sin mercado)
    merged_df = merged_df.dropna(subset=["close"])
    
    logger.success(f"Dataset final creado con {len(merged_df)} filas y {len(merged_df.columns)} columnas.")
    
    return merged_df


def run_feature_engineering() -> pd.DataFrame:
    """
    Pipeline orquestador de Feature Engineering.
    """
    logger.info("🚀 Iniciando Feature Engineering")
    
    # 1. Cargar datos crudos
    logger.info("Cargando datos desde Parquet...")
    gas_df = pd.read_parquet(DATA_CFG.DATA_DIR / "gas_prices.parquet")
    climate_df = pd.read_parquet(DATA_CFG.DATA_DIR / "climate_data.parquet")
    
    # 2. Transformar clima
    logger.info("Aplicando transformaciones climáticas...")
    climate_df = calculate_degree_days(climate_df)
    climate_df = calculate_anomalies(climate_df)
    climate_df = calculate_rolling_features(climate_df, window=7)
    
    # 3. Agregar a nivel diario
    climate_daily_df = aggregate_to_daily(climate_df)
    
    # 4. Unir con precios
    final_df = merge_and_clean(gas_df, climate_daily_df)
    
    # 5. Guardar resultado
    output_path = DATA_CFG.DATA_DIR / "final_dataset.parquet"
    final_df.to_parquet(output_path, index=False)
    logger.success(f"✓ Dataset final guardado en: {output_path}")
    
    return final_df