# YOLOv5 🚀 от Ultralytics, лицензия GPL-3.0
"""
Запуск бенчмарков YOLOv5 для всех поддерживаемых форматов экспорта

Формат                      | `export.py --include`         | Модель
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/

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
    sys.path.append(str(ROOT))  # Добавляем ROOT в PATH
# ROOT = ROOT.relative_to(Path.cwd())  # Относительный путь

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
        imgsz=640,  # Размер изображения для инференса (в пикселях)
        batch_size=1,  # Размер батча
        data=ROOT / 'data/coco128.yaml',  # Путь к файлу dataset.yaml
        device='',  # Устройство CUDA, например, 0 или 0,1,2,3 или cpu
        half=False,  # Использование FP16 для инференса с полуплавающей точностью
        test=False,  # Тестирование только экспорта
        pt_only=False,  # Тестирование только PyTorch
        hard_fail=False,  # Выбрасывание ошибки при неудаче в бенчмарке
):
    y, t = [], time.time()
    device = select_device(device)
    model_type = type(attempt_load(weights, fuse=False))  # Тип модели (DetectionModel, SegmentationModel и т.д.)
    for i, (name, f, suffix, cpu, gpu) in export.export_formats().iterrows():  # Индекс, (название, файл, расширение, CPU, GPU)
        try:
            assert i not in (9, 10), 'инференс не поддерживается'  # Edge TPU и TF.js не поддерживаются
            assert i != 5 or platform.system() == 'Darwin', 'инференс поддерживается только на macOS>=10.13'  # CoreML
            if 'cpu' in device.type:
                assert cpu, 'инференс не поддерживается на CPU'
            if 'cuda' in device.type:
                assert gpu, 'инференс не поддерживается на GPU'

            # Экспорт
            if f == '-':
                w = weights  # Формат PyTorch
            else:
                w = export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]  # Все остальные
            assert suffix in str(w), 'экспорт не удался'

            # Валидация
            if model_type == SegmentationModel:
                result = val_seg(data, w, batch_size, imgsz, plots=False, device=device, task='speed', half=half)
                metric = result[0][7]  # (box(p, r, map50, map), mask(p, r, map50, map), *loss(box, obj, cls))
            else:  # DetectionModel:
                result = val_det(data, w, batch_size, imgsz, plots=False, device=device, task='speed', half=half)
                metric = result[0][3]  # (p, r, map50, map, *loss(box, obj, cls))
            speed = result[2][1]  # Время (предобработка, инференс, постобработка)
            y.append([name, round(file_size(w), 1), round(metric, 4), round(speed, 2)])  # MB, mAP, t_inference
        except Exception as e:
            if hard_fail:
                assert type(e) is AssertionError, f'Бенчмарк --hard-fail для {name}: {e}'
            LOGGER.warning(f'ПРЕДУПРЕЖДЕНИЕ ⚠️ Неудача в бенчмарке для {name}: {e}')
            y.append([name, None, None, None])  # mAP, t_inference
        if pt_only and i == 0:
            break  # Прерываем после PyTorch

    # Вывод результатов
    LOGGER.info('\n')
    parse_opt()
    notebook_init()  # Вывод информации о системе
    c = ['Формат', 'Размер (MB)', 'mAP50-95', 'Время инференса (мс)'] if metric else ['Формат', 'Экспорт', '', '']
    py = pd.DataFrame(y, columns=c)
    LOGGER.info(f'\nБенчмарки завершены ({time.time() - t:.2f}с)')
    LOGGER.info(str(py if metric else py.iloc[:, :2]))
    if hard_fail and isinstance(hard_fail, str):
        metrics = py['mAP50-95'].array  # Значения для сравнения с порогом
        floor = eval(hard_fail)  # Минимальное значение метрики для прохождения, например, = 0.29 mAP для YOLOv5n
        assert all(x > floor for x in metrics if pd.notna(x)), f'HARD FAIL: mAP50-95 < порог {floor}'
    return py


def test(
        weights=ROOT / 'yolov5s.pt',  # Путь к весам
        imgsz=640,  # Размер изображения для инференса (в пикселях)
        batch_size=1,  # Размер батча
        data=ROOT / 'data/coco128.yaml',  # Путь к файлу dataset.yaml
        device='',  # Устройство CUDA, например, 0 или 0,1,2,3 или cpu
        half=False,  # Использование FP16 для инференса с полуплавающей точностью
        test=False,  # Тестирование только экспорта
        pt_only=False,  # Тестирование только PyTorch
        hard_fail=False,  # Выбрасывание ошибки при неудаче в бенчмарке
):
    y, t = [], time.time()
    device = select_device(device)
    for i, (name, f, suffix, gpu) in export.export_formats().iterrows():  # Индекс, (название, файл, расширение, gpu-capable)
        try:
            w = weights if f == '-' else \
                export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]  # Веса
            assert suffix in str(w), 'экспорт не удался'
            y.append([name, True])
        except Exception:
            y.append([name, False])  # mAP