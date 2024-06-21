import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

DATA_PATH = 'data/processed/data_prepared.csv'
MODEL_PATH = 'models/model.pkl'

# Чтение данных
data = pd.read_csv(DATA_PATH)

# Разделение данных на признаки и целевую переменную
X = data.drop('quality', axis=1)
y = data['quality']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Загрузка модели
model = joblib.load(MODEL_PATH)

# Оценка модели
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Model evaluation completed. MSE: {mse}')
