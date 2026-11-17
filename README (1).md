# 🌡️📈 Gas Natural AI Predictor (Henry Hub)

Pipeline cuantitativo que combina **datos climáticos** (HDD/CDD y anomalías térmicas) con **Machine Learning (LightGBM)** para explorar si el clima aporta señal predictiva sobre el retorno diario del gas natural.

> ⚠️ **Proyecto educativo y de investigación.** No constituye asesoría financiera ni una estrategia de trading lista para usar.

## ¿Qué hace?

1. **Ingesta de datos**
   - Precios históricos de gas natural (`NG=F`) desde Yahoo Finance (`yfinance`).
   - Clima diario histórico de 5 ciudades de EE.UU. (New York, Chicago, Houston, Los Angeles, Denver) desde la API gratuita de [Open-Meteo](https://open-meteo.com/).
2. **Feature Engineering**
   - Heating Degree Days (HDD) y Cooling Degree Days (CDD), base 65 °F.
   - Anomalía de HDD respecto al promedio histórico de la misma semana del año.
   - Ventanas móviles de 7 días (HDD acumulado, nieve acumulada).
   - Agregación a métricas nacionales diarias y unión con el precio.
3. **Modelado**
   - Variable objetivo: retorno porcentual del día siguiente.
   - Modelo LightGBM con división cronológica (train < 2024-01-01, test ≥ 2024-01-01).
   - Métricas: MAE, RMSE y precisión direccional. Gráfico de importancia de features.
4. **Dashboard** interactivo con Streamlit y Plotly.

## Estructura del proyecto

```
.
├── config.py               # Configuración central (tickers, ciudades, fechas, URLs)
├── data_fetcher.py         # Descarga de precios y clima
├── feature_engineering.py  # HDD, CDD, anomalías, rolling windows, merge
├── modeling.py             # Entrenamiento y evaluación de LightGBM
├── main.py                 # Orquestador del pipeline completo
├── app.py                  # Dashboard Streamlit
├── requirements.txt
└── data/                   # Se genera automáticamente (no se versiona)
```

## Instalación

Requiere Python 3.10+.

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO

python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Uso

**1. Ejecutar el pipeline completo** (descarga, features y modelo):

```bash
python main.py
```

Esto genera en `data/`:
- `gas_prices.parquet` y `climate_data.parquet` (datos crudos)
- `final_dataset.parquet` (dataset final)
- `feature_importance.png` (importancia de features)

> La descarga de clima incluye pausas para respetar los límites de Open-Meteo, así que la primera ejecución tarda algunos minutos.

**2. Lanzar el dashboard:**

```bash
streamlit run app.py
```

## Resultados

Las métricas (MAE, RMSE y precisión direccional) se imprimen en consola al terminar `main.py`. Los resultados cambian según la fecha de ejecución, ya que el rango de datos llega hasta hoy. En mercados eficientes es normal obtener una precisión direccional cercana al 50 %; interpreta cualquier resultado con cautela.

## Limitaciones conocidas

- La anomalía de HDD usa el promedio de **todo** el histórico (incluido el periodo de test), lo que introduce una fuga leve de información.
- El *early stopping* de LightGBM se evalúa sobre el conjunto de test; para una validación rigurosa conviene separar un conjunto de validación independiente.
- Solo se usan 5 ciudades y no se incluyen inventarios (EIA), pronósticos del clima ni costos de transacción.
- `NG=F` es un contrato de futuros continuo; los saltos entre vencimientos pueden afectar los retornos.

## Posibles mejoras

- Integrar datos de almacenamiento semanal de la EIA (`EIAStorageFetcher` ya está esbozado).
- Usar pronósticos climáticos (Open-Meteo Forecast) en lugar de solo clima observado.
- Validación walk-forward y backtesting con costos.
- Probar clasificación directa de dirección y más ciudades/regiones.

## Tecnologías

Python · pandas · NumPy · LightGBM · scikit-learn · yfinance · Open-Meteo · Streamlit · Plotly · loguru

## Autor

Desarrollado por **OYM** — Proyecto de Quantitative Climate Finance.

## Licencia

MIT (añade un archivo `LICENSE` si quieres aplicarla).
