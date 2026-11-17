"""
Orquestador principal del pipeline completo (Fase 1 + 2 + 3).
Ejecuta: python main.py
"""
from loguru import logger
import sys

from data_fetcher import GasPriceFetcher, ClimateDataFetcher
from feature_engineering import run_feature_engineering
from modeling import run_modeling
from config import DATA_CFG


def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )


def main():
    setup_logging()
    logger.info("🚀 Iniciando pipeline COMPLETO de Gas Natural (Fases 1, 2 y 3)")
    
    # FASE 1
    logger.info("=" * 60)
    logger.info("FASE 1: Descargando datos crudos")
    gas_fetcher = GasPriceFetcher()
    df_gas = gas_fetcher.fetch()
    df_gas.to_parquet(DATA_CFG.DATA_DIR / "gas_prices.parquet", index=False)
    
    climate_fetcher = ClimateDataFetcher()
    df_climate = climate_fetcher.fetch_all_cities()
    df_climate.to_parquet(DATA_CFG.DATA_DIR / "climate_data.parquet", index=False)
    logger.success("✓ Datos crudos guardados")
    
    # FASE 2
    logger.info("=" * 60)
    logger.info("FASE 2: Feature Engineering")
    final_df = run_feature_engineering()
    
    # FASE 3
    logger.info("=" * 60)
    logger.info("FASE 3: Modelado de IA (LightGBM)")
    run_modeling()
    
    logger.info("=" * 60)
    logger.success("🎉 ¡PIPELINE COMPLETO FINALIZADO! Revisa la carpeta 'data' para ver los resultados.")


if __name__ == "__main__":
    main()