import streamlit as st
import pandas as pd
from pickle import load
import numpy as np
import babel.numbers
import decimal
from zipfile import ZipFile
threshold = 70

@st.cache_resource
def load_model():

    zf = ZipFile('best_rf_model.zip', 'r')
    zf.extractall('')
    zf.close()
    with open("best_rf_model.pkl", "rb") as f:
        model = load(f)

    return model

@st.cache_data
def load_test_dataset():
    dataset = pd.read_csv('test_dataset.csv')

    return dataset

model = load_model()
test_data = load_test_dataset()

st.title("Analise de Dados em informacões sobre a demanda de reservas no mercado hoteleiro")
st.subheader("Integrantes", divider="gray")
st.markdown("- George da S. B. Souza")
st.markdown("- João V. dos S. Araujo")
st.markdown("- Leonardo V. W. J. da Silva")
st.markdown("- Vitor M. C. de Gouveia")

st.subheader("Lucro com Overselling", divider="green")


test_probas = model.predict_proba(test_data)

st.text_input("Insira a probabilidade mínima de cancelamento", key="cancelamento")
try:
    threshold = float(st.session_state.cancelamento) / 100.0
except:
    threshold = 70/100.0

cancel_idx = np.where(test_probas[:, 1] > threshold)[0]
canceled_samples = test_data.iloc[cancel_idx]

st.write(f'Caso realizássemos o Overselling em reservas onde nosso melhor modelo prediz cancelamento dos clientes com :blue[{threshold*100}% de certeza] (cerca de :red[{len(cancel_idx)} clientes]), nós traríamos :green[{babel.numbers.format_currency( decimal.Decimal(canceled_samples.adr.sum()), "EUR" )} de lucro] para a rede de hotéis.')

