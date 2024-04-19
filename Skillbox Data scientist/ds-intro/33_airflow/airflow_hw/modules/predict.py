# <YOUR_IMPORTS>
import pandas as pd
import dill
import json
import glob
import os

from datetime import datetime


def get_latest_model(models_dir):
  #Функция поиска последней модели в папке.

    latest_model = None
    latest_model_time = None

    for file in os.listdir(models_dir):
        if file.endswith(".pkl"):
            filename_parts = file.split("_")
            timestamp = filename_parts[2].replace(".pkl", "")
            timestamp = datetime.strptime(timestamp, "%Y%m%d%H%M")

            if latest_model_time is None or timestamp > latest_model_time:
                latest_model_time = timestamp
                latest_model = os.path.join(models_dir, file)

    return latest_model


def predict():
    path = os.environ.get('PROJECT_PATH', '/home/admin/airflow_hw')
    models_dir = f'{path}/data/models'
    
      # Получение пути к последней модели
    latest_model_path = get_latest_model(models_dir)
    
    with open(latest_model_path, 'rb') as file:
        model = dill.load(file, )
    
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    print('ok')
    # готовим путь до файлов для теста
    path_files = path + '/data/test/*json'
    # перебираем тестовые файлы из путей файлов
    for json_files_path in glob.iglob(path_files):
        with open(json_files_path) as fin:
            try:
                form = json.load(fin)
                df = pd.DataFrame.from_dict([form])
                print(df)
                pred = model.predict(df)
            except json.decoder.JSONDecodeError:
                print(f"Error decoding JSON in file: {json_files_path}")

            #x = {'car_id': df.id, 'pred': pred}
            #y = model.predict(df)

if __name__ == '__main__':
    predict()
