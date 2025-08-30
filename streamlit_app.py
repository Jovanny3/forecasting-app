import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile

st.set_page_config(page_title="Forecasting App", layout="wide")

# ----------------------------- Helpers -----------------------------
def save_uploaded_to_temp(uploaded_file):
    """Grava UploadedFile para ficheiro temporário no disco."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name

@st.cache_resource
def load_joblib_model(path):
    import joblib
    return joblib.load(path)

@st.cache_resource
def load_keras_model(path):
    from tensorflow.keras.models import load_model
    return load_model(path, compile=False)

# ----------------------------- UI -----------------------------
st.title("📊 Forecasting App — Corporativo")
st.caption("Template para deploy de modelos estatísticos (ARIMA/SARIMA/Prophet) e IA (LSTM/Keras)")

with st.sidebar:
    st.header("⚙️ Configuração")
    model_source = st.selectbox("Fonte do modelo", ["Upload", "Local Repo", "URL (S3/GDrive)"])

    uploaded_model = None
    model_path = None
    model_url = None

    if model_source == "Upload":
        uploaded_model = st.file_uploader("📂 Carregar modelo (.pkl, .joblib, .h5)",
                                          type=["pkl", "joblib", "h5", "keras"])
    elif model_source == "Local Repo":
        model_path = st.text_input("Caminho local", value="modelo.pkl")
    else:
        model_url = st.text_input("URL do modelo (ex: S3/GDrive)")

    model_type = st.selectbox("Tipo de modelo", [
        "Auto (pela extensão)", "Estatístico (ARIMA/SARIMA/Prophet)", "Keras (LSTM)"
    ])

    forecast_horizon = st.number_input("Horizonte de previsão", min_value=1, value=12)

# ----------------------------- Load Model -----------------------------
model = None
scaler = None
detected_type = None

if uploaded_model is not None:
    try:
        tmp_model_path = save_uploaded_to_temp(uploaded_model)
        ext = os.path.splitext(tmp_model_path)[1].lower()

        if ext in [".h5", ".keras"]:
            model = load_keras_model(tmp_model_path)
            detected_type = "LSTM"
        elif ext in [".pkl", ".pickle"]:
            import pickle
            with open(tmp_model_path, "rb") as f:
                model = pickle.load(f)
            detected_type = "Estatístico"
        elif ext in [".joblib"]:
            model = load_joblib_model(tmp_model_path)
            detected_type = "Estatístico"
        else:
            st.warning("Extensão não reconhecida. Define manualmente o tipo de modelo.")

    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        model = None

elif model_path:
    try:
        ext = os.path.splitext(model_path)[1].lower()
        if ext in [".h5", ".keras"]:
            model = load_keras_model(model_path)
            detected_type = "LSTM"
        elif ext in [".pkl", ".pickle"]:
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            detected_type = "Estatístico"
        elif ext in [".joblib"]:
            model = load_joblib_model(model_path)
            detected_type = "Estatístico"
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo local: {e}")

elif model_url:
    st.info("🔗 Implementar lógica de download (requests/gdown). Depois carregar localmente.")

# ----------------------------- Dados -----------------------------
st.markdown("---")
st.header("📂 Dados de Entrada")

uploaded_data = st.file_uploader("Carregar ficheiro de dados (CSV ou Excel)", type=["csv", "xlsx"])

if uploaded_data is not None:
    try:
        if uploaded_data.name.endswith(".csv"):
            df = pd.read_csv(uploaded_data)
        else:
            df = pd.read_excel(uploaded_data)

        st.success("✅ Dados carregados")
        st.write(df.head())

    except Exception as e:
        st.error(f"Erro a ler dados: {e}")
        df = None
else:
    df = None

# ----------------------------- Forecast -----------------------------
st.markdown("---")
st.header("📈 Previsões")

if model is None:
    st.warning("⚠️ Nenhum modelo carregado ainda.")
elif df is None:
    st.info("Carregue um ficheiro de dados para continuar.")
else:
    try:
        if detected_type == "Estatístico" or model_type.startswith("Estatístico"):
            if hasattr(model, "forecast"):
                preds = model.forecast(steps=int(forecast_horizon))
            elif hasattr(model, "predict"):
                preds = model.predict(len(df), len(df) + int(forecast_horizon) - 1)
            else:
                st.error("O modelo não tem métodos forecast/predict.")
                preds = None

            if preds is not None:
                preds = pd.Series(preds, name="Forecast")
                st.line_chart(preds)
                st.download_button("💾 Exportar previsões (CSV)", preds.to_csv(index=False), file_name="previsoes.csv")

        elif detected_type == "LSTM" or model_type.startswith("Keras"):
            n_steps = st.number_input("Tamanho da janela (timesteps)", min_value=1, value=12)
            values = df.iloc[:, -1].values.astype("float32")

            X = []
            for i in range(len(values) - n_steps + 1):
                X.append(values[i:i + n_steps])
            X = np.array(X)

            preds = model.predict(X)
            preds = preds[-int(forecast_horizon):].flatten()

            st.line_chart(preds)
            out = pd.DataFrame({"Forecast": preds})
            st.download_button("💾 Exportar previsões (CSV)", out.to_csv(index=False), file_name="previsoes.csv")

    except Exception as e:
        st.error(f"Erro a gerar previsões: {e}")
