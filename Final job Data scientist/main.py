# Импортируем библеотеки
import json

import dill

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Импортируем модель
with open('sber_auto_pipe.pkl', 'rb') as file:
    model = dill.load(file)


# Создаем форму данных для приема
class Form(BaseModel):
    session_id: str
    client_id: float
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str
    


# Создаем форму которая будет показываться в ответ на запрос
class Prediction(BaseModel):
    client_id: float
    Result: int


# при запросе status возвращяет "I'm OK"
@app.get('/status')
def status():
    return "I'm OK"


# при запросе version возвращяет данные о модели
@app.get('/version')
def version():
    return model['metadata']


# при запросе predict возвращает client_id и Result(предикт модели)
@app.post('/predict', response_model=Prediction, response_model_exclude_unset=True)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'client_id': form.client_id,
        'Result': int(y[0])
    }



