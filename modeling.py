"""
Módulo de Modelado de IA.
Entrena y evalúa un modelo LightGBM para predecir retornos del Gas Natural.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from loguru import logger

from config import DATA_CFG


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la variable objetivo: retorno porcentual del día siguiente.
    Usamos shift(-1) para que las features de HOY predigan el retorno de MAÑANA.
    """
    logger.info("Creando variable objetivo (retorno del día siguiente)...")
    df = df.copy()
    
    # Retorno porcentual: (Precio_mañana - Precio_hoy) / Precio_hoy
    df['target_next_day_return'] = df['close'].pct_change().shift(-1)
    
    # Dirección del mercado (1 = sube, 0 = baja) para clasificación opcional
    df['target_direction'] = (df['target_next_day_return'] > 0).astype(int)
    
    # Eliminamos la última fila porque no tiene un "día siguiente" para calcular el retorno
    df = df.dropna(subset=['target_next_day_return']).reset_index(drop=True)
    
    return df


def train_evaluate_model(df: pd.DataFrame) -> None:
    """
    Divide los datos cronológicamente, entrena LightGBM y evalúa el rendimiento.
    """
    logger.info("Preparando datos para entrenamiento cronológico...")
    
    # Definir features (X) y target (y)
    feature_cols = [
        'national_avg_hdd', 'national_max_hdd', 'national_avg_cdd', 
        'national_hdd_anomaly', 'national_max_hdd_7d', 'national_snowfall_7d',
        'volume' # El volumen también es un predictor útil
    ]
    
    X = df[feature_cols]
    y = df['target_next_day_return']
    dates = df['date']
    
    # Split CRONOLÓGICO (80% entrenamiento, 20% prueba)
    from datetime import datetime
    split_date = datetime.strptime('2024-01-01', '%Y-%m-%d').date()
    
    train_mask = dates < split_date
    test_mask = dates >= split_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info(f"Entrenamiento: {len(X_train)} filas (hasta {split_date})")
    logger.info(f"Prueba (Test): {len(X_test)} filas (desde {split_date})")
    
    # Configuración de LightGBM (optimizado para series temporales financieras)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,  # Previene overfitting
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    logger.info("Entrenando modelo LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    # Evaluación
    logger.info("Evaluando modelo en el conjunto de prueba...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Métrica de Negocio: Direccional Accuracy (¿Cuántas veces acertó la dirección?)
    pred_direction = (y_pred > 0).astype(int)
    true_direction = (y_test > 0).astype(int)
    directional_accuracy = (pred_direction == true_direction).mean()
    
    logger.success("📊 RESULTADOS DEL MODELO:")
    logger.info(f"   - MAE (Error Absoluto Medio): {mae:.4f}")
    logger.info(f"   - RMSE (Raíz del Error Cuadrático): {rmse:.4f}")
    logger.info(f"   - Direccional Accuracy: {directional_accuracy:.2%}")
    
    if directional_accuracy > 0.52:
        logger.warning("💡 ¡La precisión direccional es > 52%! Esto es prometedor para una estrategia de trading.")
    else:
        logger.info("💡 La precisión direccional es cercana al 50% (azar). Esto es normal en mercados eficientes; se necesitan más features o ajuste de hiperparámetros.")
    
    # Feature Importance (El "Alpha")
    logger.info("🔍 IMPORTANCIA DE LAS FEATURES (¿Qué mueve el precio?):")
    lgb.plot_importance(model, max_num_features=10, importance_type='gain', figsize=(10, 6))
    plt.title('Importancia de Features Climáticas y de Volumen (Gain)')
    plt.tight_layout()
    plt.savefig(DATA_CFG.DATA_DIR / "feature_importance.png", dpi=300)
    logger.success(f"✓ Gráfico de importancia guardado en: {DATA_CFG.DATA_DIR / 'feature_importance.png'}")
    
    return model


def run_modeling() -> None:
    """Orquestador de la fase de modelado."""
    logger.info("🚀 Iniciando Fase 3: Modelado de IA")
    
    # 1. Cargar dataset final
    logger.info("Cargando dataset final...")
    df = pd.read_parquet(DATA_CFG.DATA_DIR / "final_dataset.parquet")
    
    # 2. Crear target
    df = create_target_variable(df)
    
    # 3. Entrenar y evaluar
    model = train_evaluate_model(df)
    
    logger.success("🎉 Fase 3 completada exitosamente.")