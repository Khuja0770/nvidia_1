import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Функция загрузки данных
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Удаление колонки Price, если она существует
        if 'Price' in df.columns:
            df.drop(columns=['Price'], inplace=True)
        st.write("### Названия столбцов в загруженном файле:", df.columns.tolist())
        return df
    return None

st.title("Анализ видеокарт NVIDIA RTX")

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

    # Определение названия колонки для памяти
    memory_col = 'Memory Size'
    if memory_col not in data_20.columns or memory_col not in data_30.columns:
        memory_col = [col for col in data_20.columns if 'memory' in col.lower() or 'vram' in col.lower()]
        memory_col = memory_col[0] if memory_col else None

    if memory_col:
        # Сравнительные графики
        st.subheader("Сравнение CUDA Cores, GPU Models и Memory Size")

        comparison_df = pd.concat([
            data_20.assign(Series='RTX 20'),
            data_30.assign(Series='RTX 30')
        ])

        # График CUDA Cores
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Series', y='CUDA Cores', data=comparison_df, ax=ax)
        ax.set_title('Сравнение CUDA Cores между RTX 20 и RTX 30')
        st.pyplot(fig)

        # График Memory Size
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Series', y=memory_col, data=comparison_df, ci=None, ax=ax)
        ax.set_title(f'Сравнение {memory_col} между RTX 20 и RTX 30')
        st.pyplot(fig)

        # Дополнительные графики
        st.subheader("Дополнительные визуализации")

        # Boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Series', y=memory_col, data=comparison_df, ax=ax)
        ax.set_title(f'Boxplot {memory_col}')
        st.pyplot(fig)

        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='CUDA Cores', y=memory_col, hue='Series', data=comparison_df, ax=ax)
        ax.set_title(f'Scatter plot CUDA Cores vs {memory_col}')
        st.pyplot(fig)
    else:
        st.write("Не удалось найти колонку с размером памяти. Проверьте названия столбцов.")

    # Pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    series_counts = comparison_df['Series'].value_counts()
    ax.pie(series_counts, labels=series_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Распределение GPU серий')
    st.pyplot(fig)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Series', data=comparison_df, ax=ax)
    ax.set_title('Количество моделей в каждой серии')
    st.pyplot(fig)

else:
    st.write("Пожалуйста, загрузите оба CSV-файла.")
