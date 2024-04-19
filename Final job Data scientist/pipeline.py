# Импортируем нужные библиотеки

import logging
import os
import dill
import pandas as pd
import hashlib
import math

from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Создаем путь до проекта
path = os.environ.get('PROJECT_PATH', 'E:/final/final')


# Функция для удаления айди
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'session_id',
        'client_id',
    ]
    return df.drop(columns_to_drop, axis=1)


# Функция для обработки колонки device_screen_resolution
def process_screen_resolution(df, resolution_column='device_screen_resolution'):
    # Делим разрешение экрана на x и y
    df[['x_pixel', 'y_pixel']] = df[resolution_column].str.split('x', expand=True)
    df['x_pixel'] = pd.to_numeric(df['x_pixel'])
    df['y_pixel'] = pd.to_numeric(df['y_pixel'])

    # Логарифмируем значения
    df['log_pixel'] = df['x_pixel'] * df['y_pixel']
    df['log_pixel'] = df['log_pixel'].apply(lambda x: math.log(x))
    df['x_pixel'] = df['x_pixel'].apply(lambda x: min(x, 2500))
    df['y_pixel'] = df['y_pixel'].apply(lambda x: min(x, 2000))

    # Обработка выбросов
    mx = df['x_pixel'].median()
    my = df['y_pixel'].median()
    zero_pixel_rows = df['log_pixel'] == 0
    df.loc[zero_pixel_rows, 'log_pixel'] = math.log(mx * my)
    df.loc[zero_pixel_rows, 'x_pixel'] = mx
    df.loc[zero_pixel_rows, 'y_pixel'] = my

    # Умножаем x на y и заменяем значения в 'device_screen_resolution'
    df[resolution_column] = df['x_pixel'].astype(str) + 'x' + df['y_pixel'].astype(str)

    return df


# Функция для преоброзавания колонок, создание новых и удаление ненужных
def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['visit_date'], format='%Y-%m-%d') + \
                     pd.to_timedelta(df['visit_time'])

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute

    df['geo'] = df['geo_country'] + '/' + df['geo_city']

    drop_col = ['device_screen_resolution', 'visit_date', 'visit_time', 'datetime', 'geo_country', 'geo_city', 'device_model']

    return df.drop(drop_col, axis=1)


# Функция для хеширования колонок
def hash_features(df, col):
    df = df.copy()
    df[col] = df[col].applymap(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())
    return pd.DataFrame(df)


def pipeline() -> None:
    # Загрузка датафрейма
    df = pd.read_csv(f'{path}/final_job/final_data', low_memory=False)

    # Разделение на признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']

    # Создание листа с названиями колонок для преоброзавания
    scal_columns = ['visit_number', 'log_pixel', 'x_pixel', 'y_pixel', 'year', 'month', 'day', 'hour', 'minute']
    encod_columns = ['utm_medium', 'device_category', 'device_os', 'device_brand', 'device_browser', 'geo']
    hash_columns = ['utm_source', 'utm_campaign', 'utm_adcontent', 'utm_keyword']

    # Создание Пайплайна для StandardScaler
    scal_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Создание пайплайна для OneHotEncoder
    encod_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Создаем объект ColumnTransformer, который позволяет применять различные преобразования к столбцам
    column_transformer = ColumnTransformer(transformers=[
        ('numerical', scal_transformer, scal_columns),
        ('categorical', encod_transformer, encod_columns)
    ])

    # Создаем пайплайн где сначало удаялем колонки айди с помощью filter
    # После чего создаем новые колонки и удаляем ненужные с помощью feature_creator
    # Дальше с помощью hash_transformer хешируем колонки hash_columns
    # И в конце преобразуем данные с помощью column_transformer
    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('screen_resolution', FunctionTransformer(process_screen_resolution)),
        ('feature_creator', FunctionTransformer(create_features)),
        ('hash_transformer', FunctionTransformer(hash_features(X, hash_columns))),
        ('column_transformer', column_transformer)
    ])

    # Создаем список моделей машинного обучения
    models = [
        LogisticRegression(n_jobs=-1),
        RandomForestClassifier(n_jobs=-1)
    ]
    # Цикл для работы с моделями. Внутри него модель передается по пайплайну, где данные сначало преобразуются, после чего передаются в обучение
    best_score = .0
    best_pipe = None
    for model in models:

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Используя cross_val_score узнаем лучшую модель с оценкой roc_auc, которую дальше сохраняем
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        logging.info(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    logging.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')

    # Обучаем лучшую модель, даем название и данные
    best_pipe.fit(X, y)
    model_filename = f'{path}/sber_auto_pipe.pkl'

    metadata = {
        'name': 'Sber auto-subscription service prediction model',
        'author': 'Akty',
        'version': 1,
        'date': datetime.now(),
        'type': type(best_pipe.named_steps["classifier"]).__name__,
        'roc_auc': best_score
    }

    # Сохраняем модель в pkl файл
    with open(model_filename, 'wb') as file:
        dill.dump({'model': best_pipe, 'metadata': metadata}, file, recurse=True)

    logging.info(f'Model is saved as {model_filename}')


if __name__ == '__main__':
    pipeline()
