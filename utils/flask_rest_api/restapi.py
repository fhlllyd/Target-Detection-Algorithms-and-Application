# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import io

import torch
from flask import Flask, request
from PIL import Image

# Создаем экземпляр Flask-приложения
app = Flask(__name__)
# Создаем пустой словарь для хранения моделей
models = {}

# Определяем URL-шаблон для эндпоинта детекции объектов
DETECTION_URL = "/v1/object-detection/<model>"


# Регистрируем маршрут для указанного URL и методов POST
@app.route(DETECTION_URL, methods=["POST"])
# Определяем функцию обработки запросов на этот маршрут
def predict(model):
    # Проверяем, является ли метод запроса не POST
    if request.method != "POST":
        # Если метод не POST, возвращаем None
        return

    # Проверяем, есть ли файл изображения в запросе
    if request.files.get("image"):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        # Получаем файл изображения из запроса
        im_file = request.files["image"]
        # Читаем байты из файла изображения
        im_bytes = im_file.read()
        # Открываем изображение из байтов с использованием PIL
        im = Image.open(io.BytesIO(im_bytes))

        # Проверяем, есть ли запрошенная модель в словаре моделей
        if model in models:
            # Применяем модель к изображению с размером 640 (можно уменьшить до 320 для более быстрой инференции)
            results = models[model](im, size=640)
            # Возвращаем результаты в формате JSON (список записей)
            return results.pandas().xyxy[0].to_json(orient="records")


# Точка входа в программу
if __name__ == "__main__":
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    # Добавляем аргумент для порта с значением по умолчанию 5000
    parser.add_argument("--port", default=5000, type=int, help="port number")
    # Добавляем аргумент для модели (можно указать несколько моделей)
    parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    # Парсим аргументы командной строки
    opt = parser.parse_args()

    # Проходим по всем моделям из аргументов командной строки
    for m in opt.model:
        # Загружаем модель из репозитория ultralytics/yolov5
        models[m] = torch.hub.load("ultralytics/yolov5", m, force_reload=True, skip_validation=True)

    # Запускаем Flask-приложение на указанном хосте и порте
    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat