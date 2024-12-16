import streamlit as st
import pandas as pd
from model import *
from features import *

# Заголовок приложения
st.title("Прогноз дохода")
st.write("Введите параметры ниже, чтобы узнать, получает ли человек больше 50k в год.")

# Ввод параметров от пользователя
age = st.slider("Возраст", min_value=0, max_value=100, value=30, step=1)
fnlwgt = st.slider('CPS вес', min_value=19_300, max_value=1_484_700, value=189_459, step=1_000)
workclass = st.selectbox("Класс работы", workclass_data)
education = st.selectbox("Образование", education_data)
education_num = st.slider("Количество лет обучения", min_value=1, max_value=15, value=10, step=1)
marital_status = st.selectbox("Семейное положение", marital_status_data)
occupation = st.selectbox("Профессия", occupation_data)
relationship = st.selectbox("Отношения в семье", relationship_data)
race = st.selectbox("Раса", race_data)
sex = st.selectbox("Пол", sex_data)
capital_loss = st.slider('Утраченный капитал', min_value=0, max_value=5_000, value=0, step=1)
capital_gain = st.slider('Полученный капитал', min_value=0, max_value=99_999, value=0, step=1)
hours_per_week = st.slider("Часов в неделю", min_value=0, max_value=100, value=40, step=1)

# Собираем параметры в строку или словарь для передачи модели
data = {
    "age": age,
    "workclass": workclass,
    "fnlwgt": fnlwgt,
    "education": education,
    "education-num": education_num,
    "marital-status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "sex": sex,
    "capital-gain": capital_gain,
    'capital-loss': capital_loss,
    "hours-per-week": hours_per_week
}

X_test = pd.DataFrame([data])


st.write("**Параметры для модели:**")
st.dataframe(X_test)

# Кнопка для предсказания
def load_model():
    df = open_data()
    X, y = preprocess_data(df,  test=True)
    fit_and_save_model(X, y)

def generate_predict(data_df):
    try:
        train_df = open_data()
        train_X_df, _ = split_data(train_df)
        full_X_df = pd.concat((data_df, train_X_df), axis=0)
        preprocessed_X_df = preprocess_data(full_X_df, test=False)

        user_X_df = preprocessed_X_df[:1]
        prediction,prediction_proba = load_model_and_predict(user_X_df)

        st.write(beautify_results(prediction_proba,prediction))
        # st.write(f"**Прогнозируемый доход:** {prediction}")

    except AssertionError as e:
        st.error(f"Произошла ошибка: {e}")
        load_model()
        generate_predict(data_df)

# Передайте df в функцию generate_predict

if st.button("Предсказать"):
    generate_predict(X_test)
