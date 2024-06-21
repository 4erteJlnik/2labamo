import requests
import os

# URLs для скачивания данных
WHITE_WINE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
RED_WINE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
OUTPUT_DIR = 'data/raw'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Функция для скачивания и сохранения данных
def download_data(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f'Data downloaded successfully and saved to {output_path}.')
    else:
        print(f'Failed to download data from {url}.')

download_data(WHITE_WINE_URL, os.path.join(OUTPUT_DIR, 'winequality-white.csv'))
download_data(RED_WINE_URL, os.path.join(OUTPUT_DIR, 'winequality-red.csv'))
