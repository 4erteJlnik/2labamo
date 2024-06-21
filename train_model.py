import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = 'data/processed/data_prepared.csv'
MODEL_PATH = 'models/model.pkl'

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def validate_data(data):
    if 'quality' not in data.columns:
        print('Целевая переменная "quality" отсутствует в данных')
        raise ValueError('Целевая переменная "quality" отсутствует в данных')

def read_data(path):
    try:
        data = pd.read_csv(path)
        print(f'Данные успешно прочитаны из {path}')
        return data
    except Exception as e:
        print(f'Ошибка при чтении данных: {e}')
        raise

def save_model(model, path):
    try:
        joblib.dump(model, path)
        print(f'Модель успешно сохранена по пути {path}')
    except Exception as e:
        print(f'Ошибка при сохранении модели: {e}')
        raise

# Чтение данных
data = read_data(DATA_PATH)

# Валидация данных
validate_data(data)

# Разделение данных на признаки и целевую переменную
X = data.drop('quality', axis=1)
y = data['quality']

# Обработка пропущенных значений (если есть)
if X.isnull().values.any() or y.isnull().values.any():
    print('Обнаружены пропущенные значения, они будут заполнены средним значением')
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)

# Разделение данных на обучающую и тестовую выборки + перекрестная проверка
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Обучение модели
model = LinearRegression()
print('Начало обучения модели')
model.fit(X_train, y_train)
print('Модель успешно обучена')

# Сохранение модели
save_model(model, MODEL_PATH)
print('Program finished successfully')