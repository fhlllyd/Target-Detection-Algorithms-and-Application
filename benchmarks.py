# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Запуск тестов производительности YOLOv5 во всех поддерживаемых форматах экспорта

Формат                     | `export.py --include`         | Модель
---                        | ---                           | ---
PyTorch                    | -                             | yolov5s.pt
TorchScript                | `torchscript`                 | yolov5s.torchscript
ONNX                       | `onnx`                        | yolov5s.onnx
OpenVINO                   | `openvino`                    | yolov5s_openvino_model/
TensorRT                   | `engine`                      | yolov5s.engine
CoreML                     | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel      | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef        | `pb`                          | yolov5s.pb
TensorFlow Lite            | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU        | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js              | `tfjs`                        | yolov5s_web_model/

Требования:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU
    $ pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com  # TensorRT

Использование:
    $ python benchmarks.py --weights yolov5s.pt --img 640
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Корневой каталог YOLOv5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Добавление ROOT в PATH

import export
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from segment.val import run as val_seg
from utils import notebook_init
from utils.general import LOGGER, check_yaml, file_size, print_args
from utils.torch_utils import select_device
from val import run as val_det


def run(
        weights=ROOT / 'yolov5s.pt',  # Путь к весам
        imgsz=640,  # Размер изображения для вывода (пиксели)
        batch_size=1,  # Размер пакета
        data=ROOT / 'data/coco128.yaml',  # Путь к dataset.yaml
        device='',  # CUDA устройство, например, 0 или 0,1,2,3 или cpu
        half=False,  # Использовать FP16 полупrecision вывод
        test=False,  # Тестировать только экспорты
        pt_only=False,  # Тестировать только PyTorch
        hard_fail=False,  # Выбрасывать ошибку при неудачном тестировании
):
    # Инициализация переменных для результатов и времени
    y, t = [], time.time()
    device = select_device(device)  # Выбор устройства
    model_type = type(attempt_load(weights, fuse=False))  # Определение типа модели

    # Проход по всем поддерживаемым форматам экспорта
    for i, (name, f, suffix, cpu, gpu) in export.export_formats().iterrows():
        try:
            # Проверка поддержки формата на текущем устройстве
            assert i not in (9, 10), 'вывод не поддерживается'  # Edge TPU и TF.js не поддерживаются
            assert i != 5 or platform.system() == 'Darwin', 'вывод поддерживается только на macOS>=10.13'  # CoreML
            if 'cpu' in device.type:
                assert cpu, 'вывод не поддерживается на CPU'
            if 'cuda' in device.type:
                assert gpu, 'вывод не поддерживается на GPU'

            # Экспорт модели в заданный формат
            if f == '-':
                w = weights  # Формат PyTorch
            else:
                w = export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]

            # Проверка успешности экспорта
            assert suffix in str(w), 'экспорт не удался'

            # Валидация экспортированной модели
            if model_type == SegmentationModel:
                result = val_seg(data, w, batch_size, imgsz, plots=False, device=device, task='speed', half=half)
                metric = result[0][7]  # метрика качества
            else:
                result = val_det(data, w, batch_size, imgsz, plots=False, device=device, task='speed', half=half)
                metric = result[0][3]  # метрика качества
            speed = result[2][1]  # время вывода
            y.append([name, round(file_size(w), 1), round(metric, 4), round(speed, 2)])  # Сохранение результатов
        except Exception as e:
            if hard_fail:
                assert type(e) is AssertionError, f'Ошибка тестирования --hard-fail для {name}: {e}'
            LOGGER.warning(f'ПРЕДУПРЕЖДЕНИЕ ⚠️ Ошибка тестирования для {name}: {e}')
            y.append([name, None, None, None])  # Добавление результата с ошибкой
        if pt_only and i == 0:
            break  # Прерывание после PyTorch

    # Вывод результатов
    LOGGER.info('\n')
    parse_opt()
    notebook_init()  # Вывод информации о системе
    c = ['Формат', 'Размер (МБ)', 'mAP50-95', 'Время вывода (мс)']
    py = pd.DataFrame(y, columns=c)
    LOGGER.info(f'\nТесты завершены ({time.time() - t:.2f}с)')
    LOGGER.info(str(py))
    if hard_fail and isinstance(hard_fail, str):
        metrics = py['mAP50-95'].array  # Значения метрик
        floor = eval(hard_fail)  # Минимальное значение метрики для успешного тестирования
        assert all(x > floor for x in metrics if pd.notna(x)), f'КРИТИЧЕСКАЯ ОШИБКА: mAP50-95 < минимального значения {floor}'
    return py


def test(
        weights=ROOT / 'yolov5s.pt',
        imgsz=640,
        batch_size=1,
        data=ROOT / 'data/coco128.yaml',
        device='',
        half=False,
        test=False,
        pt_only=False,
        hard_fail=False,
):
    # Аналогичная логика для тестирования экспорта без оценки качества
    y, t = [], time.time()
    device = select_device(device)
    for i, (name, f, suffix, gpu) in export.export_formats().iterrows():
        try:
            w = weights if f == '-' else export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]
            assert suffix in str(w), 'экспорт не удался'
            y.append([name, True])
        except Exception:
            y.append([name, False])

    # Вывод результатов
    LOGGER.info('\n')
    parse_opt()
    notebook_init()
    py = pd.DataFrame(y, columns=['Формат', 'Экспорт'])
    LOGGER.info(f'\nЭкспорт завершен ({time.time() - t:.2f}с)')
    LOGGER.info(str(py))
    return py


def parse_opt():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='путь к весам')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='размер изображения для вывода (пиксели)')
    parser.add_argument('--batch-size', type=int, default=1, help='размер пакета')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='путь к dataset.yaml')
    parser.add_argument('--device', default='', help='CUDA устройство, например, 0 или 0,1,2,3 или cpu')
    parser.add_argument('--half', action='store_true', help='использовать FP16 полупrecision вывод')
    parser.add_argument('--test', action='store_true', help='тестировать только экспорты')
    parser.add_argument('--pt-only', action='store_true', help='тестировать только PyTorch')
    parser.add_argument('--hard-fail', nargs='?', const=True, default=False, help='Выбрасывать исключение при ошибке')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(vars(opt))
    return opt


def main(opt):
    # Запуск тестирования или основных тестов в зависимости от параметров
    test(**vars(opt)) if opt.test else run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)