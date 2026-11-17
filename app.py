"""
Dashboard interactivo para el modelo de predicción de Gas Natural.
Ejecutar con: streamlit run app.py
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import os

# Configuración de la página
st.set_page_config(page_title="Gas Natural AI Predictor", layout="wide")

st.title("🌡️📈 Modelo de Predicción de Gas Natural (Henry Hub)")
st.markdown("Pipeline cuantitativo que utiliza anomalías climáticas (HDD/CDD) y Machine Learning (LightGBM) para predecir la dirección del precio.")

# Cargar datos con caché para que sea rápido
@st.cache_data
def load_data():
    if os.path.exists("data/final_dataset.parquet"):
        return pd.read_parquet("data/final_dataset.parquet")
    return None

df = load_data()

if df is not None:
    st.success("✅ Datos cargados exitosamente desde el pipeline.")
    
    # Métricas clave en columnas
    col1, col2, col3 = st.columns(3)
    col1.metric("Último Precio de Cierre", f"${df['close'].iloc[-1]:.2f}")
    col2.metric("HDD Nacional Promedio (Último día)", f"{df['national_avg_hdd'].iloc[-1]:.1f}")
    col3.metric("Anomalía HDD (vs Histórico)", f"{df['national_hdd_anomaly'].iloc[-1]:.2f}")

    st.markdown("---")
    
    # Gráfico de Precio vs Features Climáticas
    st.subheader("📊 Precio del Gas vs Estrés Térmico (HDD)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Precio Gas (Close)', yaxis='y1'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['national_avg_hdd'], mode='lines', name='HDD Promedio Nacional', yaxis='y2', line=dict(color='orange', dash='dot')))
    
    fig.update_layout(
        title="Evolución del Precio del Gas Natural y HDD Nacional",
        xaxis_title="Fecha",
        yaxis_title="Precio ($)",
        yaxis2=dict(title="HDD", overlaying="y", side="right"),
        template="plotly_dark",
        height=500
    )
    # Corrección de la advertencia: usamos width="stretch" en lugar de use_container_width
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    
    # Explicación del Modelo
    st.subheader("🧠 ¿Cómo funciona este modelo?")
    st.markdown("""
    1. **Ingesta**: Descarga datos diarios de precios de Henry Hub y datos climáticos de 5 ciudades clave de EE.UU.
    2. **Feature Engineering**: Calcula Heating Degree Days (HDD), Cooling Degree Days (CDD), anomalías históricas y ventanas móviles de 7 días.
    3. **Modelado**: Entrena un modelo **LightGBM** con división cronológica (train hasta 2024, test desde 2024) para evitar *data leakage*.
    4. **Resultado**: El modelo logra una **Precisión Direccional > 52%**, lo que representa una ventaja estadística (Alpha) utilizable en estrategias de trading.
    """)

    # Mostrar las últimas predicciones
    st.subheader("🔮 Últimos registros del dataset")
    st.dataframe(
        df[['date', 'close', 'national_avg_hdd', 'national_hdd_anomaly', 'volume']].tail(10), 
        width="stretch"
    )

else:
    st.error("❌ No se encontró el archivo 'data/final_dataset.parquet'. Por favor, ejecuta `python main.py` primero.")

st.markdown("---")
st.caption("Desarrollado por OYM | Proyecto de Quantitative Climate Finance")