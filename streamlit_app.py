import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

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
        else:
            st.error("Tipo de modelo não reconhecido.")
            return None
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None


def stat_forecast(model, df_proc, horizon):
    """
    Tenta gerar forecast para modelos estatísticos/Prophet
    ou interpretar arrays (caso o ficheiro carregado seja apenas previsões).
    """
    # Prophet
    if "prophet" in str(type(model)).lower():
        future = model.make_future_dataframe(periods=int(horizon), freq="M")
        forecast_df = model.predict(future)[["ds", "yhat"]].tail(int(horizon))
        return forecast_df.set_index("ds")["yhat"]

    # statsmodels SARIMA
    if hasattr(model, "forecast"):
        return pd.Series(model.forecast(steps=int(horizon)))
    if hasattr(model, "get_forecast"):
        pred_obj = model.get_forecast(steps=int(horizon))
        if hasattr(pred_obj, "predicted_mean"):
            return pd.Series(pred_obj.predicted_mean)
    if hasattr(model, "predict") and callable(model.predict):
        start = len(df_proc)
        end = start + int(horizon) - 1
        return pd.Series(model.predict(start=start, end=end))

    # Se for array salvo por engano
    if isinstance(model, np.ndarray):
        arr = model
        if arr.ndim == 1 and len(arr) >= int(horizon):
            return pd.Series(arr[-int(horizon):])
        if arr.ndim == 2 and arr.shape[0] >= int(horizon):
            return pd.Series(arr[-int(horizon):, 0])
        raise RuntimeError("Array não é válido como forecast.")

    raise RuntimeError(f"Modelo não suportado: {type(model)}")


def keras_forecast(model, df_proc, scaler, window_size, horizon):
    """
    Forecast com LSTM Keras já treinada.
    """
    data = df_proc.values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    # Cria sequência inicial (última janela)
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
# Streamlit UI
# ======================
st.title("📊 Forecasting App (SARIMA, Prophet, LSTM)")

uploaded_data = st.file_uploader("📂 Carregar ficheiro Excel com a série temporal", type=["xlsx", "xls"])
uploaded_model = st.file_uploader("📂 Carregar modelo treinado (.pkl, .joblib, .h5, .keras)", type=["pkl", "joblib", "h5", "keras"])

model_type = st.selectbox("📌 Tipo de modelo:", ["Estatístico (ARIMA/SARIMA/Prophet)", "Keras (LSTM)"])
forecast_horizon = st.number_input("⏩ Quantos períodos prever:", min_value=1, max_value=60, value=12)

if uploaded_data is not None:
    df = pd.read_excel(uploaded_data)
    st.write("Pré-visualização dos dados:")
    st.write(df.head())
    # Assumindo colunas "Data" e "y"
    df["Data"] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index("Data")
    df_proc = df.iloc[:, 0]

if uploaded_model is not None:
    st.write(f"🔄 A carregar modelo ({model_type}) ...")
    model = None
    scaler = None

    if model_type == "Keras (LSTM)":
        model = load_uploaded_model(uploaded_model, "Keras (LSTM)")
        scaler_file = st.file_uploader("📂 Carregar scaler usado no treino (.pkl)", type=["pkl"])
        if scaler_file is not None:
            scaler = joblib.load(scaler_file)
    else:
        # Primeiro tenta pickle, depois joblib
        try:
            model = pickle.load(uploaded_model)
        except Exception:
            uploaded_model.seek(0)
            try:
                model = joblib.load(uploaded_model)
            except Exception as e:
                st.error(f"Erro ao carregar modelo: {e}")

    if model is not None and uploaded_data is not None:
        st.success("✅ Modelo carregado com sucesso!")

        try:
            if model_type == "Estatístico (ARIMA/SARIMA/Prophet)":
                preds_series = stat_forecast(model, df_proc, forecast_horizon)
            elif model_type == "Keras (LSTM)":
                if scaler is None:
                    st.error("⚠️ Precisas carregar também o scaler usado no treino.")
                else:
                    preds_series = keras_forecast(model, df_proc, scaler, window_size=12, horizon=forecast_horizon)

            # Se chegou aqui, plotar
            if preds_series is not None:
                st.line_chart(preds_series)
                st.download_button("💾 Descarregar previsões (CSV)",
                                   preds_series.reset_index().to_csv(index=False),
                                   file_name="previsoes.csv")
        except Exception as e:
            st.error(f"Erro a gerar previsões: {e}")
            st.write("Tipo do objeto carregado:", type(model))
            if isinstance(model, np.ndarray):
                st.write("Array shape:", model.shape)
