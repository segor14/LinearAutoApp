import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from .Utils import get_pred, show_pred, pred_session_buttoms_choice

# На написании красивой загрузки CSV я сдался, session_pred_csv в большей степени писал DeepSeek
def session_pred_csv(model_type):
    st.divider()
    st.subheader("Загрузка CSV файла")
    
    uploaded_file = st.file_uploader(
        "Загрузите CSV файл с данными об автомобилях", 
        type=['csv'],
        help="Файл должен содержать колонки: name, fuel, transmission, owner, seats",
        key="csv_uploader"
    )
    
    if uploaded_file is not None:
        try:
            df_csv = pd.read_csv(uploaded_file)
            
            required_cols = ['name', 'fuel', 'transmission', 'owner', 'seats']
            missing_cols = [col for col in required_cols if col not in df_csv.columns]
            
            if missing_cols:
                st.error(f"В загруженном файле отсутствуют обязательные колонки: {', '.join(missing_cols)}")
            else:
                st.success(f"✅ Успешно загружено {len(df_csv)} записей")
                
                with st.expander("📋 Просмотр загруженных данных"):
                    st.dataframe(df_csv.head())
                
                if st.button("🚀 Начать прогнозирование", type="primary"):
                    st.session_state['processing_mode'] = True
                    st.session_state['df_csv'] = df_csv.to_dict()
                    st.rerun()
        
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {str(e)}")

    if st.session_state.get('processing_mode', False):
        st.divider()
        st.subheader("Обработка прогнозов")
        
        df_csv = pd.DataFrame(st.session_state['df_csv'])
        
        with st.spinner("Обрабатываю данные..."):
            try:
                pred = get_pred(df_csv, model_type)
                
                results_df = df_csv.copy()
                results_df['predicted_price'] = pred
                
                st.success(f"✅ Прогнозы готовы для {len(results_df)} записей")
                st.dataframe(results_df[['name', 'predicted_price']], use_container_width=True)
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать результаты в CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )
                
            except Exception as e:
                st.error(f"Ошибка при обработке данных: {str(e)}")
                st.info("Проверьте формат данных в CSV файле")

def session_model_1():
    st.success("Вы выбрали **Модель 1**")

    st.subheader("Введите параметры интересуемого авто")

    name = st.text_input("Название с описанием", "Hyundai i20 2015-2017 Sportz 1.2")
    fuel = st.selectbox("Топливо", ['Diesel', 'Petrol', 'CNG', 'LPG'])
    transmission = st.selectbox("Коробка передач", ['Manual', 'Automatic'])
    owner = st.selectbox("Владелец", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
    seats = st.number_input("Посадочные места", min_value=2, max_value=160, value=5)

    df = pd.DataFrame({
        'name': [name],
        'fuel': [fuel],
        'transmission': [transmission],
        'owner': [owner],
        'seats': [seats]
    })

    single_prediction, csv_prediction = pred_session_buttoms_choice()

    if single_prediction:
        pred = get_pred(df, 'model1')
        show_pred(pred)
    if csv_prediction:
        st.session_state['csv_mode'] = True
        st.rerun()

    if st.session_state.get('csv_mode', False):
        session_pred_csv('model1')

def session_model_2():
    st.success("Вы выбрали **Модель 2**")

    st.subheader("Введите параметры интересуемого авто")

    name = st.text_input("Название с описанием", "Hyundai i20 2015-2017 Sportz 1.2")
    year = st.number_input("Год производства", min_value=1894, max_value=2025, value=2007)
    km_driven = st.number_input("Пробег", min_value=0, max_value=1000000, value=60000)
    age = st.number_input("Возраст (лет)", min_value=1, max_value=40, value=7)
    fuel = st.selectbox("Топливо", ['Diesel', 'Petrol', 'CNG', 'LPG'])
    seller_type = st.selectbox("Продавец", ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox("Коробка передач", ['Manual', 'Automatic'])
    owner = st.selectbox("Владелец", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
    mileage = st.text_input("Потребление", "23.4 kmpl")
    engine = st.text_input("Объем", "1248 CC")
    max_power = st.text_input("Крутящий момент", "74 bhp")
    torque = st.text_input("Мощность", "190Nm@ 2000rpm")
    seats = st.number_input("Посадочные места", min_value=2, max_value=160, value=5)

    df = pd.DataFrame({
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'age': [age],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'torque': [torque],
        'seats': [seats]
    })

    single_prediction, csv_prediction = pred_session_buttoms_choice()

    if single_prediction:
        pred = get_pred(df, 'model2')
        show_pred(pred)
    if csv_prediction:
        st.session_state['csv_mode'] = True
        st.rerun()

    if st.session_state.get('csv_mode', False):
        session_pred_csv('model2')

def session_viz_1():
    st.success("Вы выбрали **Модель 1**")
    with open('app/graphs/weights_model1.pickle', 'rb') as f:
        fig = pickle.load(f)
    st.plotly_chart(fig, use_container_width=True)

def session_viz_2():
    st.success("Вы выбрали **Модель 2**")
    with open('app/graphs/weights_model2.pickle', 'rb') as f:
        fig = pickle.load(f)
    st.plotly_chart(fig, use_container_width=True)
    
