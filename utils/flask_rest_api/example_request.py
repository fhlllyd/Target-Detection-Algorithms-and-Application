# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Perform test request
"""

import pprint

import requests

import requests
import pprint

# URL для сервиса определения объектов
DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
# Имя файла изображения, которое необходимо передать для определения объектов
IMAGE = "zidane.jpg"

# Открываем файл изображения в бинарном режиме для чтения
with open(IMAGE, "rb") as f:
    # Считываем данные изображения
    image_data = f.read()

# Отправляем POST-запрос на указанный URL с данными изображения в виде файла
# и получаем ответ в формате JSON, который преобразуем в объект Python
response = requests.post(DETECTION_URL, files={"image": image_data}).json()

# Выводим ответ в красивом формате с использованием модуля pprint
pprint.pprint(response)