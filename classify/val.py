# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 classification model on a classification dataset

Usage:
    $ bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
    $ python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate ImageNet

Usage - formats:
    $ python classify/val.py --weights yolov5s-cls.pt                 # PyTorch
                                       yolov5s-cls.torchscript        # TorchScript
                                       yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                       yolov5s-cls_openvino_model     # OpenVINO
                                       yolov5s-cls.engine             # TensorRT
                                       yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                       yolov5s-cls_saved_model        # TensorFlow SavedModel
                                       yolov5s-cls.pb                 # TensorFlow GraphDef
                                       yolov5s-cls.tflite             # TensorFlow Lite
                                       yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                       yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Полный путь к текущему скрипту
FILE = Path(__file__).resolve()
# Корневая директория YOLOv5 (2 уровня выше)
ROOT = FILE.parents[1]
# Добавление корневой директории в системный путь
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# Преобразование пути в относительный
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Импорт класса для загрузки различных типов моделей
from models.common import DetectMultiBackend
# Импорт функции для создания даталоадера классификации
from utils.dataloaders import create_classification_dataloader
# Импорт вспомогательных утилит
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_img_size, check_requirements, colorstr,
                           increment_path, print_args)
# Импорт функций для работы с устройствами PyTorch
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()  # Активация оптимизированного режима вывода
def run(
    # Путь к датасету (по умолчанию MNIST)
    data=ROOT / '../datasets/mnist',
    # Путь к весам модели
    weights=ROOT / 'yolov5s-cls.pt',
    # Размер пакета для инференса
    batch_size=128,
    # Размер изображений (пиксели)
    imgsz=224,
    # Устройство для вычислений (CUDA/CPU)
    device='',
    # Количество рабочих процессов даталоадера
    workers=8,
    # Подробный вывод
    verbose=False,
    # Директория для сохранения результатов
    project=ROOT / 'runs/val-cls',
    # Имя эксперимента
    name='exp',
    # Разрешение перезаписи директории
    exist_ok=False,
    # Использование полупрецизионной арифметики
    half=False,
    # Использование OpenCV DNN для ONNX
    dnn=False,
    # Предварительно загруженная модель
    model=None,
    # Предварительно созданный даталоадер
    dataloader=None,
    # Функция потерь
    criterion=None,
    # Объект прогресс-бара
    pbar=None,
):
    # Определение контекста (обучение или инференс)
    training = model is not None
    if training:  # Вызов из тренировочного скрипта
        # Получение устройства модели
        device, pt, jit, engine = next(model.parameters()).device, True, False, False
        # Полупрецизия только на CUDA
        half &= device.type != 'cpu'
        # Преобразование модели в полупрецизионный режим
        model.half() if half else model.float()
    else:  # Вызов как самостоятельный скрипт
        # Выбор устройства с учетом размера пакета
        device = select_device(device, batch_size=batch_size)

        # Создание директории для результатов
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Загрузка модели с поддержкой различных форматов
        model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
        # Получение параметров модели
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        # Проверка совместимости размера изображений
        imgsz = check_img_size(imgsz, s=stride)
        # Получение информации о поддержке полупрецизии
        half = model.fp16
        if engine:  # Для TensorRT модели
            batch_size = model.batch_size
        else:  # Для других форматов
            device = model.device
            if not (pt or jit):  # Для не-PyTorch моделей
                batch_size = 1
                LOGGER.info(f'Принудительный размер пакета 1 для квадратного инференса (1,3,{imgsz},{imgsz})')

        # Обработка пути к датасету
        data = Path(data)
        # Определение пути к тестовой выборке
        test_dir = data / 'test' if (data / 'test').exists() else data / 'val'
        # Создание даталоадера
        dataloader = create_classification_dataloader(
            path=test_dir,
            imgsz=imgsz,
            batch_size=batch_size,
            augment=False,
            rank=-1,
            workers=workers
        )

    # Режим оценки модели
    model.eval()
    # Списки для хранения результатов
    pred, targets, loss = [], [], 0
    # Средство измерения времени
    dt = (Profile(), Profile(), Profile())
    # Количество батчей
    n = len(dataloader)
    # Тип операции (валидация/тестирование)
    action = 'validating' if dataloader.dataset.root.stem == 'val' else 'testing'
    # Формирование описания прогресс-бара
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"
    # Создание прогресс-бара
    bar = tqdm(dataloader, desc=desc, total=n, disable=training, bar_format=TQDM_BAR_FORMAT, position=0)

    # Активация автоматического масштабирования амплитуды
    with torch.cuda.amp.autocast(enabled=device.type != 'cpu'):
        for images, labels in bar:
            with dt[0]:  # Замер времени предобработки
                images, labels = images.to(device, non_blocking=True), labels.to(device)

            with dt[1]:  # Замер времени инференса
                y = model(images)

            with dt[2]:  # Замер времени постобработки
                pred.append(y.argsort(1, descending=True)[:, :5])
                targets.append(labels)
                if criterion:
                    loss += criterion(y, labels)

    # Средняя потеря
    loss /= n
    # Объединение результатов
    pred, targets = torch.cat(pred), torch.cat(targets)
    # Вычисление точности
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)
    top1, top5 = acc.mean(0).tolist()

    if pbar:  # Обновление прогресс-бара
        pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"
    if verbose:  # Подробный вывод
        LOGGER.info(f"{'Class':>24}{'Images':>12}{'top1_acc':>12}{'top5_acc':>12}")
        LOGGER.info(f"{'all':>24}{targets.shape[0]:>12}{top1:>12.3g}{top5:>12.3g}")
        for i, c in model.names.items():
            aci = acc[targets == i]
            top1i, top5i = aci.mean(0).tolist()
            LOGGER.info(f"{c:>24}{aci.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}")

        # Вывод производительности
        t = tuple(x.t / len(dataloader.dataset.samples) * 1E3 for x in dt)
        shape = (1, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}' % t)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    return top1, top5, loss


def parse_opt():
    # Создание парсера аргументов командной строки
    parser = argparse.ArgumentParser()

    # Параметр пути к датасету
    parser.add_argument('--data', type=str, default=ROOT / '../datasets/mnist',
                        help='Путь к датасету')

    # Параметр пути к весам модели (может быть несколько)
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'yolov5s-cls.pt',
                        help='Путь к файлам с весами модели (model.pt)')

    # Параметр размера пакета данных
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Размер пакета (batch size)')

    # Параметр размера изображения для инференса
    parser.add_argument('--imgsz', '--img', '--img-size', type=int,
                        default=224,
                        help='Размер изображения для инференса (пиксели)')

    # Параметр устройства для вычислений
    parser.add_argument('--device', default='',
                        help='Устройство CUDA (например, 0 или 0,1,2,3) или cpu')

    # Параметр максимального количества рабочих потоков даталоадера
    parser.add_argument('--workers', type=int, default=8,
                        help='Максимальное количество рабочих потоков для даталоадера (на каждый RANK в режиме DDP)')

    # Параметр для подробного вывода
    parser.add_argument('--verbose', nargs='?', const=True, default=True,
                        help='Подробный вывод информации')

    # Параметр директории для сохранения результатов
    parser.add_argument('--project', default=ROOT / 'runs/val-cls',
                        help='Директория для сохранения результатов (project/name)')

    # Параметр имени эксперимента
    parser.add_argument('--name', default='exp',
                        help='Имя эксперимента (добавляется к пути project/name)')

    # Параметр разрешения перезаписи существующей директории
    parser.add_argument('--exist-ok', action='store_true',
                        help='Разрешить использовать существующую директорию без инкрементирования имени')

    # Параметр использования полупрецизионной арифметики
    parser.add_argument('--half', action='store_true',
                        help='Использовать полупрецизионную арифметику FP16')

    # Параметр использования OpenCV DNN для ONNX
    parser.add_argument('--dnn', action='store_true',
                        help='Использовать OpenCV DNN для инференса с моделями ONNX')

    # Парсинг аргументов и вывод их значений
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    # Проверка требуемых зависимостей
    check_requirements(exclude=('tensorboard', 'thop'))
    # Запуск основной логики с переданными параметрами
    run(**vars(opt))


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    opt = parse_opt()
    # Вызов главной функции
    main(opt)