import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.markdown("<h1 style='color: whitesmoke;'>Обзор данных видеокарт NVIDIA и AMD</h1>", unsafe_allow_html=True)

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

set_background_image("https://hyperpc.ru/images/support/articles/nvidia-vs-amd/nvidia-vs-amd-banner_webp.jpg")

uploaded_files = st.file_uploader("Загрузите CSV-файлы с видеокартами", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        st.markdown(f"<p style='color: white;'>Файл {file.name} успешно загружен!</p>", unsafe_allow_html=True)
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='latin-1')
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True).dropna(axis=1, how='all')

    if 'GPU Model' in combined_df.columns:
        search_query = st.text_input("Введите название видеокарты:")
        if search_query:
            result = combined_df[combined_df["GPU Model"].astype(str).str.contains(search_query, case=False, na=False)]
            st.write("### Найденные результаты:")
            st.write(result) if not result.empty else st.write("Видеокарта не найдена.")

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
            st.write(comparison_df)
            st.write("### Анализ выбранных видеокарт")
            st.write("#### Первые 5 строк", comparison_df.head())
            st.write("#### Размер данных", comparison_df.shape)
            
            if "Price (then)" in comparison_df.columns:
                fig, ax = plt.subplots()
                comparison_df.boxplot(column="Price (then)", ax=ax)
                st.pyplot(fig)
            
            if "CUDA Cores" in comparison_df.columns and "Boost Clock (MHz)" in comparison_df.columns:
                fig, ax = plt.subplots()
                ax.scatter(comparison_df["CUDA Cores"], comparison_df["Boost Clock (MHz)"], color='blue')
                ax.set_xlabel("CUDA Cores")
                ax.set_ylabel("Boost Clock (MHz)")
                st.pyplot(fig)
            
            if "Memory Size" in comparison_df.columns:
                fig, ax = plt.subplots()
                comparison_df["Memory Size"].hist(bins=10, ax=ax, color='green')
                ax.set_xlabel("Memory Size (GB)")
                ax.set_ylabel("Count")
                st.pyplot(fig)

    if st.session_state.show_analysis:
        st.write("### Анализ данных")
        st.write("#### Первые 5 строк", combined_df.head())
        st.write("#### Размер данных", combined_df.shape)
        
        column_to_count = st.selectbox("Выберите колонку для value_counts()", combined_df.columns)
        st.write(combined_df[column_to_count].value_counts())
        
        sort_column = st.selectbox("Выберите колонку для сортировки", combined_df.columns)
        st.write(combined_df.sort_values(by=sort_column))
        
        group_column = st.selectbox("Выберите колонку для группировки", combined_df.columns)
        grouped_df = combined_df.groupby(group_column).mean(numeric_only=True)
        st.write(grouped_df)
        
        st.write("### Визуализация данных")
        
        if "Price (then)" in combined_df.columns:
            fig, ax = plt.subplots()
            combined_df.boxplot(column="Price (then)", ax=ax)
            st.pyplot(fig)
        
        if "CUDA Cores" in combined_df.columns and "Boost Clock (MHz)" in combined_df.columns:
            fig, ax = plt.subplots()
            ax.scatter(combined_df["CUDA Cores"], combined_df["Boost Clock (MHz)"], color='blue')
            ax.set_xlabel("CUDA Cores")
            ax.set_ylabel("Boost Clock (MHz)")
            st.pyplot(fig)
        
        if "Memory Size" in combined_df.columns:
            fig, ax = plt.subplots()
            combined_df["Memory Size"].hist(bins=10, ax=ax, color='green')
            ax.set_xlabel("Memory Size (GB)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        
        pie_column = st.selectbox("Выберите колонку для pie chart", combined_df.columns)
        if combined_df[pie_column].nunique() > 0:
            fig, ax = plt.subplots()
            combined_df[pie_column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)
