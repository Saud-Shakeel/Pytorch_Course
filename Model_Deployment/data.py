import zipfile
import requests
from pathlib import Path
import os

def download_data(source_url, data_folder_name):
    DATA_PATH = Path('Data')
    IMG_FOLDER = DATA_PATH/data_folder_name
    if IMG_FOLDER.is_dir():
        print(f'{IMG_FOLDER} Already exists, Skipping the creation part...')
    else:
        print(f'Creating the {IMG_FOLDER}...')
        IMG_FOLDER.mkdir(parents=True, exist_ok=True)

    ZIP_PATH = DATA_PATH/f'{data_folder_name}.zip'    
    if ZIP_PATH.is_dir():
        print(f'{ZIP_PATH} Already exists, skipping download...')
    else:
        print(f'Downloading {ZIP_PATH}...')
        with open(ZIP_PATH, 'wb') as f:
            req = requests.get(source_url)
            f.write(req.content)

    with zipfile.ZipFile(ZIP_PATH, 'r') as zipRef:
        zipRef.extractall(IMG_FOLDER)
    
    os.remove(ZIP_PATH)

download_data("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", 'pizza_steak_sushi')