# app_streamlit_template.py
# Template Streamlit app para deploy de modelos estatísticos (ARIMA/SARIMA/Prophet)
# e modelos IA (LSTM/Keras). Personalize as funções de preprocessamento.

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile

st.set_page_config(page_title="Previsão - Template", layout="wide")

st.title("📈 App de Previsão — Template (ARIMA / LSTM)")

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
    """Exemplo genérico: garantir índice DateTime e coluna de série.
    Ajuste esta função conforme a pipeline usada durante treino.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.set_index(df.columns[0], inplace=True)
        except Exception:
            pass
    return df


def preprocess_for_lstm(df, n_steps=12):
    """Exemplo simples: transforma valores em janelas (sem scaling).
    Substitua pelo scaler e normalização que usou em treino.
    """
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
        "Upload (recomendado)", "No repositório (já incluído)", "URL / Cloud (S3 / Drive / HF)"
    ])

    uploaded_model = None
    model_path = None
    model_url = None

    if model_source == "Upload (recomendado)":
        uploaded_model = st.file_uploader("Carregar ficheiro do modelo (.pkl, .joblib, .h5)",
                                         type=["pkl", "joblib", "h5", "pickle"]) 
    elif model_source == "No repositório (já incluído)":
        model_path = st.text_input("Caminho local (ex: modelo.pkl)", value="modelo.pkl")
    else:
        model_url = st.text_input("URL directo (S3 / GDrive / HF Raw)")

    model_type = st.selectbox("Tipo de modelo (ajuste se necessário)", [
        "Auto (por extensão)", "Estatístico (ARIMA/SARIMA/Prophet)", "LSTM / Keras"
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
            detected_type = "Estatístico"
        except Exception as e:
            st.error(f"Erro ao carregar modelo estatístico: {e}")
    elif ext in [".h5", ".keras"]:
        try:
            model = load_keras_model(model_file_path)
            detected_type = "LSTM"
        except Exception as e:
            st.error(f"Erro ao carregar modelo Keras: {e}")
    else:
        st.warning("Extensão não reconhecida — selecione manualmente o tipo de modelo.")

elif model_path:
    ext = os.path.splitext(model_path)[1].lower()
    try:
        if ext in [".pkl", ".joblib", ".pickle"]:
            model = load_joblib_model(model_path)
            detected_type = "Estatístico"
        elif ext in [".h5", ".keras"]:
            model = load_keras_model(model_path)
            detected_type = "LSTM"
    except Exception as e:
        st.error(f"Erro ao carregar modelo local: {e}")

elif model_url:
    st.info("Se escolher URL, carregue o ficheiro no runtime (requests/gdown) e depois carregue localmente. Use st.secrets para credenciais.)")

# ----------------------------- Main -----------------------------
st.markdown("---")
st.header("Dados de entrada")


uploaded_data = st.file_uploader("Carregar ficheiro de série temporal", type=["csv", "xlsx", "xls"])


if uploaded_data is not None:
    
    if uploaded_data.name.endswith(".csv"):
        df = pd.read_csv(uploaded_data)
    else:
        df = pd.read_excel(uploaded_data)

    st.write("Preview dos dados:", df.head())

    df_proc = preprocess_for_stat_model(df.copy())

    if model is None:
        st.warning("Nenhum modelo carregado ainda. Faça upload do ficheiro do modelo na barra lateral.")
    else:
        st.success(f"Modelo carregado. Tipo detectado: {detected_type}")

        

        if detected_type == "Estatístico" or model_type == "Estatístico (ARIMA/SARIMA/Prophet)":
            try:
                if hasattr(model, "forecast"):
                    preds = model.forecast(steps=int(forecast_horizon))
                elif hasattr(model, "predict"):
                    preds = model.predict(len(df_proc), len(df_proc) + int(forecast_horizon) - 1)
                else:
                    st.error("O modelo estatístico não expõe métodos forecast/predict padrão. Ajuste a parte de previsão.")
                    preds = None

                if preds is not None:
                    preds = pd.Series(preds, name="prediction")
                    st.line_chart(preds)
                    out = preds.reset_index(drop=True)
                    st.download_button("Descarregar previsões (CSV)", out.to_csv(index=False), file_name="previsoes.csv")
            except Exception as e:
                st.error(f"Erro a gerar previsões com modelo estatístico: {e}")

        elif detected_type == "LSTM" or model_type == "LSTM / Keras":
            try:
                n_steps = st.number_input("Tamanho da janela (timesteps) para LSTM", min_value=1, value=12)
                X = preprocess_for_lstm(df_proc, n_steps=n_steps)
                st.write("Shape X:", X.shape)
                preds = model.predict(X)
                preds = preds[-int(forecast_horizon):].flatten()
                st.line_chart(preds)
                out = pd.DataFrame({"prediction": preds})
                st.download_button("Descarregar previsões (CSV)", out.to_csv(index=False), file_name="previsoes.csv")
            except Exception as e:
                st.error(f"Erro a gerar previsões com modelo LSTM: {e}")

else:
    st.info("Carregue um CSV para começar. Ajuste a função de preprocessamento ao formato do seu dataset.")

# ----------------------------- FIM -----------------------------

# Observações:
# - Personalize `preprocess_for_stat_model` e `preprocess_for_lstm` para coincidir com a pipeline que usou no Colab.
# - Para ficheiros grandes (>100MB) use Git LFS ou armazene o modelo num serviço (S3, HF, GDrive) e descarregue no runtime.
# - Use st.secrets para credenciais sensíveis (S3 keys, tokens).
