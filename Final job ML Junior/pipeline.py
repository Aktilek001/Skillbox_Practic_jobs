import os
import pandas as pd
import numpy as np
import tqdm
import pickle
import logging

from datetime import datetime
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer
from functools import partial


import warnings
warnings.filterwarnings('ignore')

# Устанавливаем уровень логирования
logging.basicConfig(level=logging.INFO)


def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0, num_parts_to_read: int = 2, columns=None, verbose=False) -> pd.DataFrame:
    """
    Читает набор данных в формате Parquet из локальной директории.

    Аргументы:
    - path_to_dataset: str - Путь к директории с набором данных.
    - start_from: int - Начальный индекс части набора данных для чтения.
    - num_parts_to_read: int - Количество частей набора данных для чтения.
    - columns: List[str] - Список столбцов для чтения из набора данных.
    - verbose: bool - Флаг для вывода информации о чтении.

    Возвращает:
    - pd.DataFrame: Результат объединения прочитанных частей набора данных.
    """
    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset) if filename.startswith('train')])
    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in tqdm.tqdm_notebook(chunks, desc="Reading dataset with pandas"):
        chunk = pd.read_parquet(chunk_path, columns=columns)
        res.append(chunk)
    return pd.concat(res).reset_index(drop=True)


def prepare_transactions_dataset(path_to_dataset: str, num_parts_to_preprocess_at_once: int = 1, num_parts_total: int=50, save_to_path=None, verbose: bool=False):
    """
    Подготавливает набор данных, выполняя препроцессинг и агрегацию.

    Аргументы:
    - path_to_dataset: str - Путь к директории с набором данных.
    - num_parts_to_preprocess_at_once: int - Количество частей набора данных для препроцессинга за один раз.
    - num_parts_total: int - Общее количество частей набора данных.
    - save_to_path: str - Путь для сохранения препроцессированных данных.
    - verbose: bool - Флаг для вывода информации о препроцессинге.

    Возвращает:
    - pd.DataFrame: Результат объединения препроцессированных частей набора данных.
    """
    data_frame = pd.DataFrame()  # Создаем пустой датафрейм
    for step in tqdm.tqdm_notebook(range(0, num_parts_total, num_parts_to_preprocess_at_once), desc="Transforming transactions data"):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once, verbose=verbose)
        # Здесь должен быть препроцессинг данных

        # Сохранение препроцессированных данных
        if save_to_path:
            block_as_str = str(step)
            if len(block_as_str) == 1:
                block_as_str = '00' + block_as_str
            else:
                block_as_str = '0' + block_as_str
            transactions_frame.to_parquet(os.path.join(save_to_path, f'processed_chunk_{block_as_str}.parquet'))
        data_frame = pd.concat([data_frame, transactions_frame])  # Обновляем датафрейм
    return data_frame.reset_index(drop=True)  # Возвращаем объединенный датафрейм


class CountAggregator(object):
    """
    Класс для агрегации данных и кодирования категориальных признаков.
    """
    
    def __init__(self):
        """
        Инициализация объекта CountAggregator.
        """
        self.encoded_features = None  # Инициализация списка закодированных признаков
        
    def __extract_count_aggregations(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Производит агрегацию и кодирование категориальных признаков входного датафрейма.

        Аргументы:
        - data_frame: pd.DataFrame - Входной датафрейм.

        Возвращает:
        - pd.DataFrame: Результат агрегации и кодирования.
        """
        # Отбрасываем идентификационные и лишние колонки
        feature_columns = list(data_frame.columns.values)
        feature_columns.remove("id")
        feature_columns.remove("rn")

        # Применяем One-Hot Encoding
        dummies = pd.get_dummies(data_frame[feature_columns], columns=feature_columns)
        dummy_features = dummies.columns.values
        
        # Объединяем закодированные признаки с исходным датафреймом
        ohe_features = pd.concat([data_frame, dummies], axis=1)
        ohe_features = ohe_features.drop(columns=feature_columns)
        
        # Агрегируем данные по идентификатору
        features = ohe_features.groupby("id")[dummy_features].sum().reset_index(drop=False)
        return features
        
    def fit_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Производит агрегацию и кодирование категориальных признаков входного датафрейма.

        Аргументы:
        - data_frame: pd.DataFrame - Входной датафрейм.

        Возвращает:
        - pd.DataFrame: Результат агрегации и кодирования.
        """
        features = self.__extract_count_aggregations(data_frame)
        features.fillna(np.uint8(0), inplace=True)
        dummy_features = list(features.columns.values)
        dummy_features.remove("id")
        self.encoded_features = dummy_features
        return features
    
    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Производит агрегацию и кодирование категориальных признаков входного датафрейма.

        Аргументы:
        - data_frame: pd.DataFrame - Входной датафрейм.

        Возвращает:
        - pd.DataFrame: Результат агрегации и кодирования.
        """
        features = self.__extract_count_aggregations(data_frame)
        features.fillna(np.uint8(0), inplace=True)
        dummy_features = list(features.columns.values)
        dummy_features.remove("id")
        for col in self.encoded_features:
            if col not in dummy_features:
                features[col] = np.uint8(0)
        return features


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет столбец 'id' из датафрейма.

    Аргументы:
    - df: pd.DataFrame - Входной датафрейм.

    Возвращает:
    - pd.DataFrame: Датафрейм без столбца 'id'.
    """
    columns_to_drop = ['id']
    return df.drop(columns_to_drop, axis=1)


def merge_targets(data_frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет датафрейм с целями с исходным датафреймом по столбцу 'id'.

    Аргументы:
    - data_frame: pd.DataFrame - Исходный датафрейм.
    - targets: pd.DataFrame - Датафрейм с целевыми значениями.

    Возвращает:
    - pd.DataFrame: Результат объединения.
    """
    return pd.merge(data_frame, targets, on='id', how='inner')


def split_features_and_target(df):
    """
    Разделяет признаки и целевую переменную.

    Аргументы:
    - df: pd.DataFrame - Исходный датафрейм.

    Возвращает:
    - tuple: Кортеж, содержащий признаки и целевую переменную.
    """
    X = df.drop('flag', axis=1)
    y = df['flag']
    
    X = X.astype('int8')
    y = y.astype('int8')
    return X, y


def pipeline() -> None:
    """
    Основная функция для выполнения всего процесса обработки данных и построения модели.

    Аргументы:
    - None

    Возвращает:
    - None
    """
    # Загрузка данных
    path = 'traindata/'
    data_frame = prepare_transactions_dataset(path_to_dataset=path, num_parts_to_preprocess_at_once=2, num_parts_total=2, save_to_path='traindata/')
    targets = pd.read_csv('train_target.csv')
    
    logging.info("Data loaded successfully.")
    
    count_aggregator = CountAggregator()

    preprocessor = Pipeline(steps=[
        ('data_aggregation', FunctionTransformer(count_aggregator.fit_transform)),
        ('data_merge', FunctionTransformer(partial(merge_targets, targets=targets))),
        ('data_filtering', FunctionTransformer(filter_data)),
        ('splitter', FunctionTransformer(split_features_and_target))
    ])

    # Получаем признаки X и целевую переменную y из преобразованного набора данных
    X, y = preprocessor.transform(data_frame)

    # Создаем модель машинного обучения
    model = LGBMClassifier(n_estimators=1000, max_depth=20, learning_rate=0.01, num_leaves=34, reg_lambda=1, class_weight='balanced', random_state=42, device='gpu')

    # Создаем пайплайн, объединяющий препроцессор и модель
    pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

    # Оцениваем модель с использованием кросс-валидации
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', error_score='raise')
    print("Cross-validation scores (ROC AUC):", cv_scores)
    print("Mean ROC AUC:", cv_scores.mean())
    print("Standard deviation of ROC AUC:", cv_scores.std())
    
    # Обучаем модель на всех данных
    model.fit(X, y)
    model_filename = 'lgbm_model.pkl'

    # Сохраняем модель и метаданные
    metadata = {
        'name': 'Модель кредитного риск-менеджмента',
        'author': 'Akty',
        'version': 1,
        'date': datetime.now(),
        'type': 'LGBMClassifier',
        'roc_auc': cv_scores.mean()
    }
    with open(model_filename, 'wb') as file:
        pickle.dump({'model': pipe, 'metadata': metadata}, file)
    print(f'Model is saved as {model_filename}')


# Запуск главной функции пайплайна
if __name__ == "__main__":
    pipeline()

