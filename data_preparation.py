import pandas as pd
import os

INPUT_DIR = 'data/raw'
OUTPUT_PATH = 'data/processed/data_prepared.csv'

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Чтение данных
white_wine = pd.read_csv(os.path.join(INPUT_DIR, 'winequality-white.csv'), sep=';')
red_wine = pd.read_csv(os.path.join(INPUT_DIR, 'winequality-red.csv'), sep=';')

# Добавление метки цвета вина
white_wine['color'] = 0  # Белое вино
red_wine['color'] = 1    # Красное вино

# Объединение датасетов
data = pd.concat([white_wine, red_wine], ignore_index=True)

# Пример обработки данных (удаление пустых значений, нормализация и т.д.)
data = data.dropna()
data = (data - data.min()) / (data.max() - data.min())

# Сохранение обработанных данных
data.to_csv(OUTPUT_PATH, index=False)
print('Data prepared successfully.')
