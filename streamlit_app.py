# app_streamlit.py
# App Streamlit para deploy de modelos estatísticos (SARIMA/Prophet) e IA (LSTM)

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile

st.set_page_config(page_title="Previsão", layout="wide")

st.title("📈 App de Previsão — SARIMA / Prophet / LSTM")

# ----------------------------- Helpers -----------------------------
@st.cache_resource
def load_joblib_model(path):
    import joblib
    return joblib.load(path)

@st.cache_resource
def load_keras_model(path):
    from tensorflow.keras.models import load_model
    return load_model(path)

def save_uploaded_file(uploaded_file):
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def preprocess_for_stat_model(df):
    """Garante índice datetime e série numérica"""
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.set_index(df.columns[0], inplace=True)
        except Exception:
            pass
    return df

def preprocess_for_lstm(df, n_steps=12):
    """Transforma série em janelas de entrada (não aplica scaler ainda)"""
    values = df.values.astype("float32")
    X = []
    for i in range(len(values) - n_steps + 1):
        X.append(values[i:i + n_steps])
    X = np.array(X)
    return X

# ----------------------------- UI: Sidebar -----------------------------
with st.sidebar:
    st.header("Configuração")
    model_source = st.selectbox("Onde está o modelo?", [
        "Upload", "No repositório", "URL / Cloud"
    ])

    uploaded_model = None
    model_path = None
    model_url = None

    if model_source == "Upload":
        uploaded_model = st.file_uploader("Carregar ficheiro do modelo (.pkl, .joblib, .h5)", 
                                         type=["pkl", "joblib", "h5", "pickle"])
    elif model_source == "No repositório":
        model_path = st.text_input("Caminho local (ex: modelo.pkl)", value="modelo.pkl")
    else:
        model_url = st.text_input("URL directo (S3 / GDrive / HF Raw)")

    model_type = st.selectbox("Tipo de modelo", [
        "Auto", "SARIMA / ARIMA", "Prophet", "LSTM"
    ])

    forecast_horizon = st.number_input("Horizonte de previsão (passos)", min_value=1, value=12)

# ----------------------------- Carregar modelo -----------------------------
model = None
detected_type = None

if uploaded_model is not None:
    model_file_path = save_uploaded_file(uploaded_model)
    ext = os.path.splitext(uploaded_model.name)[1].lower()
    if ext in [".pkl", ".joblib", ".pickle"]:
        try:
            model = load_joblib_model(model_file_path)
            detected_type = "SARIMA/ARIMA"
        except Exception as e:
            st.error(f"Erro ao carregar modelo estatístico: {e}")
    elif ext in [".h5", ".keras"]:
        try:
            model = load_keras_model(model_file_path)
            detected_type = "LSTM"
        except Exception as e:
            st.error(f"Erro ao carregar modelo Keras: {e}")
    else:
        st.warning("Extensão não reconhecida.")

elif model_path:
    ext = os.path.splitext(model_path)[1].lower()
    try:
        if ext in [".pkl", ".joblib", ".pickle"]:
            model = load_joblib_model(model_path)
            detected_type = "SARIMA/ARIMA"
        elif ext in [".h5", ".keras"]:
            model = load_keras_model(model_path)
            detected_type = "LSTM"
    except Exception as e:
        st.error(f"Erro ao carregar modelo local: {e}")

elif model_url:
    st.info("Ainda não implementado: carregar modelo a partir de URL.")

# ----------------------------- Main -----------------------------
st.markdown("---")
st.header("Dados de entrada")

uploaded_data = st.file_uploader("Carregar ficheiro de dados (CSV/Excel)", type=["csv", "xlsx", "xls"])

if uploaded_data is not None:
    if uploaded_data.name.endswith(".csv"):
        df = pd.read_csv(uploaded_data)
    else:
        df = pd.read_excel(uploaded_data)

    st.write("Preview dos dados:", df.head())

    df_proc = preprocess_for_stat_model(df.copy())

    if model is None:
        st.warning("Nenhum modelo carregado ainda.")
    else:
        st.success(f"Modelo carregado. Tipo detectado: {detected_type or model_type}")

        try:
            # ---------------- SARIMA / ARIMA ----------------
            if detected_type == "SARIMA/ARIMA" or model_type == "SARIMA / ARIMA":
                preds = model.forecast(steps=int(forecast_horizon))
                preds = pd.Series(preds, name="prediction")
                st.line_chart(preds)
                st.download_button("Descarregar previsões (CSV)", preds.to_csv(index=False), file_name="previsoes_sarima.csv")

            # ---------------- Prophet ----------------
            elif model_type == "Prophet" or "prophet" in str(type(model)).lower():
                future = model.make_future_dataframe(periods=int(forecast_horizon), freq="M")
                preds = model.predict(future)[["ds", "yhat"]].tail(int(forecast_horizon))
                st.line_chart(preds.set_index("ds")["yhat"])
                out = preds.rename(columns={"ds": "date", "yhat": "prediction"})
                st.download_button("Descarregar previsões (CSV)", out.to_csv(index=False), file_name="previsoes_prophet.csv")

            # ---------------- LSTM ----------------
            elif detected_type == "LSTM" or model_type == "LSTM":
                n_steps = st.number_input("Tamanho da janela (timesteps)", min_value=1, value=12)
                X = preprocess_for_lstm(df_proc, n_steps=n_steps)
                preds = model.predict(X)
                preds = preds[-int(forecast_horizon):].flatten()
                st.line_chart(preds)
                out = pd.DataFrame({"prediction": preds})
                st.download_button("Descarregar previsões (CSV)", out.to_csv(index=False), file_name="previsoes_lstm.csv")

        except Exception as e:
            st.error(f"Erro a gerar previsões: {e}")

else:
    st.info("Carregue um ficheiro CSV ou Excel para começar.")
