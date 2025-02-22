import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стилей
sns.set(style="whitegrid")

# Функция загрузки данных
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Заголовок приложения
st.title("📊 Анализ видеокарт NVIDIA RTX")

# Загрузка файлов CSV
uploaded_file_20 = st.file_uploader("Загрузите CSV для RTX 20 Series", type=["csv"])
uploaded_file_30 = st.file_uploader("Загрузите CSV для RTX 30 Series", type=["csv"])

# Загрузка данных
data_20 = load_data(uploaded_file_20)
data_30 = load_data(uploaded_file_30)

if data_20 is not None and data_30 is not None:
    st.subheader("Обзор данных")
    st.write("### RTX 20 Series")
    st.write(data_20.head())
    st.write("Форма данных:", data_20.shape)
    st.write("### RTX 30 Series")
    st.write(data_30.head())
    st.write("Форма данных:", data_30.shape)

    # Описание данных
    st.subheader("Статистическое описание")
    st.write("### RTX 20 Series")
    st.write(data_20.describe())
    st.write("### RTX 30 Series")
    st.write(data_30.describe())
else:
    st.write("Пожалуйста, загрузите оба CSV-файла.")
