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

# Загрузка файлов
uploaded_file_20 = st.file_uploader("Загрузите CSV для RTX 20 Series", type=["csv"])
uploaded_file_30 = st.file_uploader("Загрузите CSV для RTX 30 Series", type=["csv"])

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

    # Объединение данных для сравнения
    data_20['Series'] = 'RTX 20'
    data_30['Series'] = 'RTX 30'
    combined_data = pd.concat([data_20, data_30], ignore_index=True)

    # Функция для отображения сравнительных графиков
    st.subheader("📈 Сравнение RTX 20 и RTX 30")

    # Сравнение CUDA Cores
    if 'GPU Model' in combined_data.columns and 'CUDA Cores' in combined_data.columns:
        st.write("### 🔹 Сравнение CUDA Cores")
        plt.figure(figsize=(14, 8))
        sns.barplot(x='GPU Model', y='CUDA Cores', hue='Series', data=combined_data, palette="muted")
        plt.xticks(rotation=45)
        plt.xlabel("GPU Model")
        plt.ylabel("CUDA Cores")
        plt.title("Сравнение количества ядер CUDA между RTX 20 и RTX 30")
        st.pyplot(plt)
        plt.close()

    # Сравнение Memory Size
    if 'GPU Model' in combined_data.columns and 'Memory Size (GB)' in combined_data.columns:
        st.write("### 🔹 Сравнение Memory Size")
        plt.figure(figsize=(14, 8))
        sns.barplot(x='GPU Model', y='Memory Size (GB)', hue='Series', data=combined_data, palette="coolwarm")
        plt.xticks(rotation=45)
        plt.xlabel("GPU Model")
        plt.ylabel("Memory Size (GB)")
        plt.title("Сравнение объёма памяти между RTX 20 и RTX 30")
        st.pyplot(plt)
        plt.close()

else:
    st.write("Пожалуйста, загрузите оба CSV-файла.")
