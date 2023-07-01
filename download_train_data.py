import requests
from io import BytesIO
from PIL import Image
import os
import csv
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

urls_path = 'csc-hackathon-2023-lunua-task/train.csv'
save_path = 'data/images/'
test_path = 'data/train.csv'

urls = pd.read_csv(urls_path)

saved = os.listdir(save_path)

def load_and_save_image(url):
    try:
        name = os.path.basename(url)
        if not name in saved:
            response = requests.get(url)
            image_data = response.content
            image = Image.open(BytesIO(image_data))
            image.save(save_path + name)
            
        return name
    except Exception as e:
        return None


with open(test_path, 'a') as f:
    writer = csv.writer(f)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for _, (url1, url2, equal) in tqdm(urls.iterrows(), total=len(urls)):
            future1 = executor.submit(load_and_save_image, url1)
            future2 = executor.submit(load_and_save_image, url2)

            name1 = future1.result()
            name2 = future2.result()

            if name1 and name2:
                writer.writerow([name1, name2, equal])
