import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# ======================
# Configuração inicial
# ======================
st.set_page_config(page_title="Forecasting App", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #F7F9FB;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .card {
        padding: 1.2rem;
        border-radius: 0.8rem;
        background-color: #ffffff;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ======================
# Funções utilitárias
# ======================
def load_uploaded_model(uploaded_file, model_type):
    try:
        if model_type == "Keras (LSTM)":
            return load_model(uploaded_file, compile=False)
        elif model_type == "Pickle":
            return pickle.load(uploaded_file)
        elif model_type == "Joblib":
            return joblib.load(uploaded_file)
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return None

def stat_forecast(model, df_proc, horizon):
    if "prophet" in str(type(model)).lower():
        future = model.make_future_dataframe(periods=int(horizon), freq="M")
        forecast_df = model.predict(future)[["ds", "yhat"]].tail(int(horizon))
        return forecast_df.set_index("ds")["yhat"]
    if hasattr(model, "forecast"):
        return pd.Series(model.forecast(steps=int(horizon)))
    if hasattr(model, "get_forecast"):
        pred_obj = model.get_forecast(steps=int(horizon))
        return pd.Series(pred_obj.predicted_mean)
    if hasattr(model, "predict") and callable(model.predict):
        start = len(df_proc)
        end = start + int(horizon) - 1
        return pd.Series(model.predict(start=start, end=end))
    if isinstance(model, np.ndarray):
        return pd.Series(model[-int(horizon):].flatten())
    raise RuntimeError(f"Modelo não suportado: {type(model)}")

def keras_forecast(model, df_proc, scaler, window_size, horizon):
    data = df_proc.values.reshape(-1, 1)
    scaled_data = scaler.transform(data)
    last_window = scaled_data[-window_size:]
    preds_scaled = []
    for _ in range(int(horizon)):
        X_input = last_window.reshape(1, window_size, 1)
        pred = model.predict(X_input, verbose=0)
        preds_scaled.append(pred[0][0])
        last_window = np.append(last_window[1:], pred)[-window_size:]
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
    return pd.Series(preds.flatten())


# ======================
# Layout do App
# ======================
st.title("📊 Forecasting Dashboard")
st.markdown("### Previsão com SARIMA, Prophet e LSTM — Interface Profissional")

tab1, tab2, tab3 = st.tabs(["📂 Carregar Dados", "⚙️ Carregar Modelo", "📊 Previsões & Comparação"])

# ====== TAB 1 ======
with tab1:
    st.markdown("<div class='card'><h4>📂 Upload dos Dados</h4></div>", unsafe_allow_html=True)
    uploaded_data = st.file_uploader("Carregar ficheiro Excel", type=["xlsx", "xls"])
    if uploaded_data:
        df = pd.read_excel(uploaded_data)
        st.dataframe(df.head(), use_container_width=True)

        df["Data"] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index("Data")
        df_proc = df.iloc[:, 0]

# ====== TAB 2 ======
with tab2:
    st.markdown("<div class='card'><h4>⚙️ Upload do Modelo</h4></div>", unsafe_allow_html=True)
    model_type = st.selectbox("Tipo de modelo:", ["Estatístico (ARIMA/SARIMA/Prophet)", "Keras (LSTM)"])
    uploaded_model = st.file_uploader("Carregar modelo treinado", type=["pkl", "joblib", "h5", "keras"])
    scaler = None

    if uploaded_model:
        if model_type == "Keras (LSTM)":
            model = load_uploaded_model(uploaded_model, "Keras (LSTM)")
            scaler_file = st.file_uploader("Carregar scaler usado no treino (.pkl)", type=["pkl"])
            if scaler_file:
                scaler = joblib.load(scaler_file)
        else:
            try:
                model = pickle.load(uploaded_model)
            except:
                uploaded_model.seek(0)
                model = joblib.load(uploaded_model)
        st.success("✅ Modelo carregado com sucesso!")

# ====== TAB 3 ======
with tab3:
    st.markdown("<div class='card'><h4>📊 Geração de Previsões</h4></div>", unsafe_allow_html=True)
    forecast_horizon = st.slider("⏩ Horizonte de previsão (meses)", 1, 60, 12)

    if 'model' in locals() and 'df_proc' in locals():
        try:
            if model_type == "Estatístico (ARIMA/SARIMA/Prophet)":
                preds_series = stat_forecast(model, df_proc, forecast_horizon)
            elif model_type == "Keras (LSTM)" and scaler is not None:
                preds_series = keras_forecast(model, df_proc, scaler, 12, forecast_horizon)
            else:
                preds_series = None

            if preds_series is not None:
                fig = px.line(preds_series, title="📈 Previsões")
                st.plotly_chart(fig, use_container_width=True)

                st.download_button("💾 Descarregar Previsões (CSV)",
                                   preds_series.reset_index().to_csv(index=False),
                                   file_name="previsoes.csv")
        except Exception as e:
            st.error(f"❌ Erro a gerar previsões: {e}")
