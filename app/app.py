import streamlit as st
import numpy as np
import pandas as pd
import pickle
from DataPreparation import preparation_for_model_1, preparation_for_model_2
from AppUtils import *

st.title("Сервис придумывания стоимости Вашего коня")
tab1, tab2, tab3 = st.tabs(["Получить прогноз", "Визуализация обучающих данных", "Визуализация весов модели"])  # Вкладки
metric1 = [0.93, 0.39, 0.16]
metric2 = [0.95, 0.44, 0.23]

with tab1:
    if "model" not in st.session_state:
        st.header("Выберите модель прогноза")
        st.markdown(" - **Модель 1** основана на Ridge-регрессии с использованием только категориальных переменных")
        st.markdown(" - **Модель 2** - авторская модель с комплексным анализом всех переменных и их более " \
                    "глубокой обработкой. В основе лежит Ridge-регрессия с гиперпараметрами _Модели 1_")
        
        col1, col2 = st.columns(2)
        with col1:
            metrics_card("Модель 1", list(map(str, metric1)), get_diff(metric1, metric2))
            select1 = st.button("Выбрать модель 1", key="btn1", use_container_width=True)
            if select1:
                st.session_state['model'] = "model_1"
                st.rerun()
        with col2:
            metrics_card("Модель 2",  list(map(str, metric2)), get_diff(metric2, metric1))
            select2 = st.button("Выбрать модель 2", key="btn2", use_container_width=True)
            if select2:
                st.session_state['model'] = "model_2"
                st.rerun()
        st.markdown("*__WMSPE__ - авторская метрика, придуманная засыпающим Егором (мной), которая отражает взвешенное " \
                    "среднее квадратов процентных ошибок прогноза (чем меньше, тем лучше)")
        
    elif st.session_state['model'] == 'model_1':
        session_model_1()

        if st.button("← Выбрать другую модель", key="back_from_model1"):
            del st.session_state["model"]
            st.rerun()
    
    elif st.session_state['model']=='model_2':
        session_model_2()

        if st.button("← Выбрать другую модель", key="back_from_model2"):
            del st.session_state["model"]
            st.rerun()

with tab2:
    with st.expander("📈 Pairplot для обучающей выборки", expanded=False):
        paiplot_block()

    with st.expander("📈 Корреляционная карта для обучающей выборки", expanded=False):
        heatmap_block()

    with st.expander("📈 Корреляционная карта Phik для обучающей выборки", expanded=False):
        phik_block()

    with st.expander("📈 Проверка выбросов", expanded=False):
        boxplot_block()

    with st.expander("📈 Диаграмма рассеяния для цены и количества владельцев", expanded=False):
        scatter_block()
    
    with st.expander("📈 Графики распределения для числовых признаков", expanded=False):
        distribution_block()

with tab3:
    if "model_w" not in st.session_state:
        st.header("Выберите модель прогноза")
        col1, col2 = st.columns(2)
        with col1:
            select1 = st.button("Выбрать модель 1", key="weights1", use_container_width=True)
            if select1:
                st.session_state['model_w'] = "model_1"
                st.rerun()
        with col2:
            select2 = st.button("Выбрать модель 2", key="weights2", use_container_width=True)
            if select2:
                st.session_state['model_w'] = "model_2"
                st.rerun()

    elif st.session_state['model_w'] == 'model_1':
        if st.button("← Выбрать другую модель", key="back_from_weights1"):
            del st.session_state["model_w"]
            st.rerun()
        session_viz_1()
    
    elif st.session_state['model_w']=='model_2':
        if st.button("← Выбрать другую модель", key="back_from_weights2"):
            del st.session_state["model_w"]
            st.rerun()
        session_viz_2()