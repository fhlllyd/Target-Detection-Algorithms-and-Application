# YOLOv5 🚀 от Ultralytics, лицензия GPL-3.0
"""
Проверка обученной модели YOLOv5 для задачи детекции объектов на наборе данных

Использование:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Поддерживаемые форматы:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime или OpenCV DNN с --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (только для macOS)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

# Импорт необходимых модулей
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Настройка путей к файлам и каталогам
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Корневой каталог проекта YOLOv5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Добавляем корневой каталог в список путей для поиска модулей
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Относительный путь к корневому каталогу

# Импорт внутренних модулей YOLOv5
from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


# Сохранение результатов детекции в текстовый файл
def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # Коэффициенты нормализации [h, w, h, w]
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # Преобразование в нормализованный формат xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # Формирование строки для записи в файл
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")  # Запись строки в файл


# Сохранение результатов детекции в JSON-файл
def save_one_json(predn, jdict, path, class_map):
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem  # Идентификатор изображения
    box = xyxy2xywh(predn[:, :4])  # Преобразование координат bounding box из xyxy в xywh
    box[:, :2] -= box[:, 2:] / 2  # Преобразование в координаты верхнего левого угла
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],  # Маппинг класса
                "bbox": [round(x, 3) for x in b],  # Округление координат bounding box до 3 знаков
                "score": round(p[4], 5),  # Округление значения confidence до 5 знаков
            }
        )


# Обработка одного батча детекций
def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)  # Матрица правильных предсказаний
    iou = box_iou(labels[:, 1:], detections[:, :4])  # Вычисление IoU между истинными и предсказанными bounding box
    correct_class = labels[:, 0:1] == detections[:, 5]  # Проверка соответствия классов
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # Нахождение индексов, удовлетворяющих условиям IoU и класса
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # Формирование массива соответствий
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]  # Сортировка по убыванию IoU
                matches = matches[
                    np.unique(matches[:, 1], return_index=True)[1]
                ]  # Удаление дубликатов предсказаний
                matches = matches[
                    np.unique(matches[:, 0], return_index=True)[1]
                ]  # Удаление дубликатов истинных значений
            correct[matches[:, 1].astype(int), i] = True  # Запись результатов в матрицу правильных предсказаний
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


# Основная функция для валидации модели
@smart_inference_mode()
def run(
    data,  # Путь к файлу конфигурации набора данных
    weights=None,  # Путь к файлу весов модели
    batch_size=32,  # Размер батча
    imgsz=640,  # Размер изображения для инференса (в пикселях)
    conf_thres=0.001,  # Порог уверенности
    iou_thres=0.6,  # Порог IoU для NMS
    max_det=300,  # Максимальное количество детекций на изображение
    task="val",  # Тип задачи (валидация, тестирование и т.д.)
    device="",  # Устройство для вычислений (GPU или CPU)
    workers=8,  # Количество рабочих процессов для загрузчика данных
    single_cls=False,  # Флаг для одноклассовой задачи
    augment=False,  # Флаг для использования аугментации при инференсе
    verbose=False,  # Флаг для вывода подробной информации
    save_txt=False,  # Флаг для сохранения результатов в текстовый файл
    save_hybrid=False,  # Флаг для сохранения гибридных результатов (метки + предсказания) в текстовый файл
    save_conf=False,  # Флаг для сохранения значений уверенности в текстовый файл
    save_json=False,  # Флаг для сохранения результатов в JSON-файл
    project=ROOT / "runs/val",  # Каталог для сохранения результатов
    name="exp",  # Название эксперимента
    exist_ok=False,  # Флаг для перезаписи существующего каталога
    half=True,  # Флаг для использования полуточного режима (FP16)
    dnn=False,  # Флаг для использования OpenCV DNN для инференса ONNX
    model=None,  # Модель (если уже загружена)
    dataloader=None,  # Загрузчик данных (если уже создан)
    save_dir=Path(""),  # Каталог для сохранения результатов (если уже определен)
    plots=True,  # Флаг для построения графиков
    callbacks=Callbacks(),  # Обратные вызовы
    compute_loss=None,  # Функция для вычисления потерь (если требуется)
):
    # Инициализация/загрузка модели и настройка устройства
    training = model is not None  # Проверка, является ли вызов из режима обучения
    if training:  # Если вызов из режима обучения (из train.py)
        device, pt, jit, engine = (
            next(model.parameters()).device,
            True,
            False,
            False,
        )  # Получение устройства, на котором находится модель (PyTorch)
        half &= device.type != "cpu"  # Полуточный режим поддерживается только на CUDA-устройствах
        model.half() if half else model.float()
    else:  # Если вызов напрямую
        device = select_device(device, batch_size=batch_size)  # Выбор устройства для вычислений

        # Создание каталогов для сохранения результатов
        save_dir = increment_path(
            Path(project) / name, exist_ok=exist_ok
        )  # Создание уникального пути для сохранения результатов
        (save_dir / "labels" if save_txt else save_dir).mkdir(
            parents=True, exist_ok=True
        )  # Создание каталога для сохранения текстовых файлов с метками

        # Загрузка модели
        model = DetectMultiBackend(
            weights, device=device, dnn=dnn, data=data, fp16=half
        )
        stride, pt, jit, engine = (
            model.stride,
            model.pt,
            model.jit,
            model.engine,
        )
        imgsz = check_img_size(imgsz, s=stride)  # Проверка размера изображения
        half = model.fp16  # Поддержка полуточного режима
        if engine:
            batch_size = model.batch_size  # Размер батча для модели TensorRT
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # Размер батча по умолчанию для моделей, не основанных на PyTorch
                LOGGER.info(
                    f"Принудительное использование размера батча 1 для квадратного инференса (1,3,{imgsz},{imgsz}) для не-PyTorch моделей"
                )

        # Загрузка данных
        data = check_dataset(data)  # Проверка файла конфигурации набора данных

    # Настройка модели
    model.eval()  # Перевод модели в режим оценки
    cuda = device.type != "cpu"  # Проверка использования CUDA
    is_coco = (
        isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")
    )  # Проверка, является ли набор данных COCO
    nc = 1 if single_cls else int(data["nc"])  # Количество классов
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # Вектор IoU для вычисления mAP@0.5:0.95
    niou = iouv.numel()  # Количество уровней IoU

    # Создание загрузчика данных
    if not training:
        if pt and not single_cls:  # Проверка совместимости весов и набора данных
            ncm = model.model.nc
            assert (
                ncm == nc
            ), f"{weights} ({ncm} классов) обучена на другом наборе данных, чем переданный ({nc} классов). Передайте корректную комбинацию --weights и --data."
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # Прогрев модели
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # Параметры для инференса
        task = task if task in ("train", "val", "test") else "val"  # Корректировка типа задачи
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

    # Инициализация переменных
    seen = 0  # Количество обработанных изображений
    confusion_matrix = ConfusionMatrix(nc=nc)  # Матрица混淆
    names = (
        model.names if hasattr(model, "names") else model.module.names
    )  # Названия классов
    if isinstance(names, (list, tuple)):  # Преобразование в словарь, если необходимо
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))  # Маппинг классов для COCO
    s = (
        "%22s" + "%11s" * 6
    ) % ("Класс", "Изображения", "Экземпляры", "Точность", "Полнота", "mAP50", "mAP50-95")  # Формат вывода
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )  # Инициализация метрик
    dt = Profile(), Profile(), Profile()  # Профилировщики времени
    loss = torch.zeros(3, device=device)  # Инициализация потерь
    jdict, stats, ap, ap_class = [], [], [], []  # Структуры для хранения результатов
    callbacks.run("on_val_start")  # Вызов обратного вызова
    pbar = tqdm(
        dataloader, desc=s, bar_format=TQDM_BAR_FORMAT
    )  # Создание индикатора прогресса

    # Обработка батчей данных
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run("on_val_batch_start")  # Вызов обратного вызова
        with dt[0]:  # Время предобработки
            if cuda:
                im = im.to(device, non_blocking=True)  # Перенос изображений на устройство
                targets = targets.to(device)
            im = im.half() if half else im.float()  # Преобразование в полуточный или точный формат
            im /= 255  # Нормализация изображений
            nb, _, height, width = im.shape  # Размеры батча и изображений

        # Инференс
        with dt[1]:  # Время инференса
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Вычисление потерь
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # Накопление потерь

        # Немаксимальное подавление (NMS)
        targets[:, 2:] *= torch.tensor(
            (width, height, width, height), device=device
        )  # Преобразование координат целевых bounding box в пиксели
        lb = (
            [targets[targets[:, 0] == i, 1:] for i in range(nb)]
            if save_hybrid
            else []
        )  # Подготовка данных для сохранения гибридных результатов
        with dt[2]:  # Время постобработки
            preds = non_max_suppression(
                preds,
                conf_thres,
                iou_thres,
                labels=lb,
                multi_label=True,
                agnostic=single_cls,
                max_det=max_det,
            )

        # Вычисление метрик
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]  # Истинные метки для текущего изображения
            nl, npr = labels.shape[0], pred.shape[0]  # Количество истинных и предсказанных объектов
            path, shape = Path(paths[si]), shapes[si][0]  # Путь к изображению и его форма
            correct = torch.zeros(
                npr, niou, dtype=torch.bool, device=device
            )  # Матрица правильных предсказаний
            seen += 1  # Увеличение счетчика обработанных изображений

            if npr == 0:  # Если нет предсказаний
                if nl:  # Если есть истинные метки
                    stats.append(
                        (correct, *torch.zeros((2, 0), device=device), labels[:, 0])
                    )  # Добавление данных в статистику
                    if plots:
                        confusion_matrix.process_batch(
                            detections=None, labels=labels[:, 0]
                        )  # Обновление матрицы混淆
                continue

            # Обработка предсказаний
            if single_cls:
                pred[:, 5] = 0  # Для одноклассовой задачи устанавливаем класс 0
            predn = pred.clone()
            scale_boxes(
                im[si].shape[1:], predn[:, :4], shape, shapes[si][1]
            )  # Масштабирование bounding box до исходного размера изображения

            # Оценка предсказаний
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # Преобразование истинных bounding box в формат xyxy
                scale_boxes(
                    im[si].shape[1:], tbox, shape, shapes[si][1]
                )  # Масштабирование истинных bounding box
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # Создание массива истинных меток с масштабированными bounding box
                correct = process_batch(predn, labelsn, iouv)  # Обработка батча предсказаний
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)  # Обновление матрицы混淆
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # Добавление данных в статистику

            # Сохранение результатов
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # Сохранение в JSON
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])  # Вызов обратного вызова

        # Построение изображений с результатами
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # Построение изображений с метками
            plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # Построение изображений с предсказаниями
        callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)  # Вызов обратного вызова

    # Вычисление метрик
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # Преобразование статистики в numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            *stats, plot=plots, save_dir=save_dir, names=names
        )  # Вычисление AP для каждого класса
        ap50, ap = ap[:, 0], ap.mean(1)  # Вычисление mAP@0.5 и mAP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()  # Вычисление средних метрик
    nt = np.bincount(
        stats[3].astype(int), minlength=nc
    )  # Подсчет количества истинных объектов для каждого класса

    # Вывод результатов
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # Формат вывода
    LOGGER.info(pf % ("Все", seen, nt.sum(), mp, mr, map50, map))  # Вывод итоговых метрик
    if nt.sum() == 0:
        LOGGER.warning(
            f"Предупреждение ⚠️ в наборе данных {task} не найдено меток, невозможно вычислить метрики без меток"
        )  # Вывод предупреждения, если нет меток

    # Вывод метрик для каждого класса
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))  # Вывод метрик для каждого класса

    # Вывод скорости обработки
    t = tuple(x.t / seen * 1E3 for x in dt)  # Среднее время обработки одного изображения в миллисекундах
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)  # Размер входных данных
        LOGGER.info(
            f"Скорость: %.1fмс предобработка, %.1fмс инференс, %.1fмс NMS на изображение при размере {shape}"
            % t
        )  # Вывод скорости обработки

    # Построение графиков
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))  # Построение матрицы混淆
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)  # Вызов обратного вызова

    # Сохранение результатов в JSON
    if save_json and len(jdict):
        w = (
            Path(weights[0] if isinstance(weights, list) else weights).stem
            if weights is not None
            else ""
        )  # Имя файла весов без расширения
        anno_json = str(
            Path(data.get("path", "../coco")) / "annotations/instances_val2017.json"
        )  # Путь к файлу аннотаций COCO
        pred_json = str(save_dir / f"{w}_predictions.json")  # Путь к файлу с предсказаниями
        LOGGER.info(f"\nОценка mAP с использованием pycocotools... сохранение {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)  # Сохранение предсказаний в JSON

        try:  # Оценка mAP с использованием pycocotools
            check_requirements("pycocotools")
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # Загрузка аннотаций
            pred = anno.loadRes(pred_json)  # Загрузка предсказаний
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [
                    int(Path(x).stem) for x in dataloader.dataset.im_files
                ]  # Установка идентификаторов изображений для оценки
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # Обновление метрик mAP
        except Exception as e:
            LOGGER.info(f"pycocotools не может быть запущен: {e}")

    # Возврат результатов
    model.float()  # Преобразование модели в точный формат
    if not training:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} меток сохранено в {save_dir / 'labels'}"
            if save_txt
            else ""
        )  # Вывод количества сохраненных меток
        LOGGER.info(f"Результаты сохранены в {colorstr('bold', save_dir)}{s}")  # Вывод пути сохранения результатов
    maps = np.zeros(nc) + map  # Инициализация mAP для каждого класса
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]  # Заполнение mAP для каждого класса
    return (
        (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()),
        maps,
        t,
    )  # Возврат метрик, mAP и времени обработки


# Парсинг аргументов командной строки
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default=ROOT / "data/coco128.yaml", help="путь к файлу конфигурации набора данных"
    )
    parser.add_argument(
        "--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="путь к файлу весов модели"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="размер батча")
    parser.add_argument(
        "--imgsz", "--img", "--img-size", type=int, default=640, help="размер изображения для инференса (в пикселях)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.001, help="порог уверенности"
    )
    parser.add_argument("--iou-thres", type=float, default=0.6, help="порог IoU для NMS")
    parser.add_argument(
        "--max-det", type=int, default=300, help="максимальное количество детекций на изображение"
    )
    parser.add_argument("--task", default="val", help="тип задачи (обучение, валидация, тестирование, оценка скорости или исследование)")
    parser.add_argument("--device", default="", help="устройство для вычислений (номер CUDA-устройства или CPU)")
    parser.add_argument("--workers", type=int, default=8, help="максимальное количество рабочих процессов для загрузчика данных")
    parser.add_argument("--single-cls", action="store_true", help="обработка набора данных как одноклассового")
    parser.add_argument("--augment", action="store_true", help="использование аугментации при инференсе")
    parser.add_argument("--verbose", action="store_true", help="подробный вывод метрик для каждого класса")
    parser.add_argument("--save-txt", action="store_true", help="сохранение результатов в текстовый файл")
    parser.add_argument("--save-hybrid", action="store_true", help="сохранение гибридных результатов (метки + предсказания) в текстовый файл")
    parser.add_argument("--save-conf", action="store_true", help="сохранение значений уверенности в текстовый файл")
    parser.add_argument("--save-json", action="store_true", help="сохранение результатов в JSON-файл")
    parser.add_argument("--project", default=ROOT / "runs/val", help="каталог для сохранения результатов")
    parser.add_argument("--name", default="exp", help="название эксперимента")
    parser.add_argument("--exist-ok", action="store_true", help="разрешить перезапись существующего каталога")
    parser.add_argument("--half", action="store_true", help="использование полуточного режима (FP16) для инференса")
    parser.add_argument("--dnn", action="store_true", help="использование OpenCV DNN для инференса ONNX")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # Проверка файла конфигурации набора данных
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


# Основная функция
def main(opt):
    check_requirements(exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # Обычный режим работы
        if opt.conf_thres > 0.001:
            LOGGER.info(
                f"Предупреждение ⚠️ порог уверенности {opt.conf_thres} > 0.001 приводит к некорректным результатам"
            )
        if opt.save_hybrid:
            LOGGER.info(
                "Предупреждение ⚠️ --save-hybrid приводит к высокой mAP из гибридных меток, а не из предсказаний"
            )
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"
        if opt.task == "speed":  # Тестирование скорости
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # Исследование зависимости между скоростью и точностью
            for opt.weights in weights:
                f = (
                    f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"
                )  # Имя файла для сохранения результатов
                x, y = list(range(256, 1536 + 128, 128)), []  # Диапазон размеров изображений и список для сохранения результатов
                for opt.imgsz in x:  # Перебор размеров изображений
                    LOGGER.info(f"\nЗапуск {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # Сохранение результатов и времени обработки
                np.savetxt(f, y, fmt="%10.4g")  # Сохранение результатов в файл
            os.system("zip -r study.zip study_*.txt")  # Архивирование результатов
            plot_val_study(x=x)  # Построение графика зависимости между скоростью и точностью


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)