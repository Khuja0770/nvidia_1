import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")  # Устанавливаем широкий режим отображения

st.markdown("<h1 style='color: white;'>Обзор данных видеокарт NVIDIA и AMD</h1>", unsafe_allow_html=True)

# Функция для установки фонового изображения и стилизации элементов
def set_background_image(image_url: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_url}) no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Установить фон (замени URL на нужный)
set_background_image("https://static.tildacdn.com/tild3962-6630-4333-b837-643036336263/AMD-vs-Nvidia.jpeg")

# Загрузка файлов (множественный выбор)
uploaded_files = st.file_uploader("Загрузите CSV-файлы с видеокартами", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    dfs = []  # Список для хранения загруженных DataFrame
    for file in uploaded_files:
        st.markdown(f"<p style='color: white;'>Файл {file.name} успешно загружен!</p>", unsafe_allow_html=True)
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='latin-1')
        dfs.append(df)

    # Объединение всех загруженных файлов в один DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.dropna(axis=1, how='all')  # Удаление пустых колонок

    # Поле поиска
    if 'GPU Model' in combined_df.columns:
        search_query = st.text_input("Введите название видеокарты:")

        if search_query:
            result = combined_df[combined_df["GPU Model"].astype(str).str.contains(search_query, case=False, na=False)]
            if not result.empty:
                st.write("### Найденные результаты:")
                st.dataframe(result)  # Используем dataframe для лучшего отображения
            else:
                st.write("Видеокарта не найдена.")

    col1, col2 = st.columns(2)

    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False

    with col1:
        if st.session_state.show_analysis:
            if st.button("Скрыть Анализ Данных"):
                st.session_state.show_analysis = False
        else:
            if st.button("Показать Анализ Данных"):
                st.session_state.show_analysis = True

    if "GPU Model" in combined_df.columns:
        with col2:
            selected_gpus = st.multiselect("Выберите до 5 видеокарт для сравнения:", combined_df["GPU Model"].unique(), max_selections=5)
        
        if selected_gpus:
            comparison_df = combined_df[combined_df["GPU Model"].isin(selected_gpus)]
            st.write("### Сравнение видеокарт")
            st.dataframe(comparison_df)  # Используем dataframe для лучшего отображения
            
            st.write("### Анализ выбранных видеокарт")
            st.write("#### Первые 5 строк")
            st.dataframe(comparison_df.head())
            st.write("#### Размер данных", comparison_df.shape)
            
            # Scatter plot
            if "CUDA Cores" in comparison_df.columns and "Boost Clock (MHz)" in comparison_df.columns:
                fig = px.scatter(comparison_df, x="CUDA Cores", y="Boost Clock (MHz)", color="GPU Model")
                st.plotly_chart(fig)
            
            # Histogram
            if "Memory Size" in comparison_df.columns:
                fig = px.histogram(comparison_df, x="Memory Size", nbins=10, title="Распределение памяти")
                st.plotly_chart(fig)

    if st.session_state.show_analysis:
        st.write("### Анализ данных")
        st.write("#### Первые 5 строк")
        st.dataframe(combined_df.head())
        st.write("#### Размер данных", combined_df.shape)
        
        # Количество значений
        column_to_count = st.selectbox("Выберите колонку для value_counts()", combined_df.columns)
        st.write(combined_df[column_to_count].value_counts())
        
        # Сортировка данных
        sort_column = st.selectbox("Выберите колонку для сортировки", combined_df.columns)
        st.dataframe(combined_df.sort_values(by=sort_column))
        
        # Группировка данных
        group_column = st.selectbox("Выберите колонку для группировки", combined_df.columns)
        grouped_df = combined_df.groupby(group_column).mean(numeric_only=True)
        st.dataframe(grouped_df)
        
        # Визуализация данных
        st.write("### Визуализация данных")
        
        # Scatter plot
        if "CUDA Cores" in combined_df.columns and "Boost Clock (MHz)" in combined_df.columns:
            fig = px.scatter(combined_df, x="CUDA Cores", y="Boost Clock (MHz)", color="GPU Model")
            st.plotly_chart(fig)
        
        # Histogram
        if "Memory Size" in combined_df.columns:
            fig = px.histogram(combined_df, x="Memory Size", nbins=10, title="Распределение памяти")
            st.plotly_chart(fig)
        
        # Pie chart
        pie_column = st.selectbox("Выберите колонку для pie chart", combined_df.columns)
        if combined_df[pie_column].nunique() > 0:
            pie_data = combined_df[pie_column].value_counts().reset_index()
            pie_data.columns = [pie_column, "Count"]
            fig = px.pie(pie_data, names=pie_column, values="Count", title="Распределение значений")
            st.plotly_chart(fig)
