# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 segment model on a segment dataset

Usage:
    $ bash data/scripts/get_coco.sh --val --segments  # download COCO-segments val split (1G, 5000 images)
    $ python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate COCO-segments

Usage - formats:
    $ python segment/val.py --weights yolov5s-seg.pt                 # PyTorch
                                      yolov5s-seg.torchscript        # TorchScript
                                      yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s-seg_openvino_label     # OpenVINO
                                      yolov5s-seg.engine             # TensorRT
                                      yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                      yolov5s-seg_saved_model        # TensorFlow SavedModel
                                      yolov5s-seg.pb                 # TensorFlow GraphDef
                                      yolov5s-seg.tflite             # TensorFlow Lite
                                      yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                      yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Получаем абсолютный путь к текущему файлу
FILE = Path(__file__).resolve()
# Определяем корневую директорию YOLOv5 (родительская директория второго уровня)
ROOT = FILE.parents[1]  # Директория корня YOLOv5
# Если путь к корневой директории не в списке путей Python, добавляем его
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Добавляем ROOT в PATH
# Преобразуем путь к корневой директории в относительный путь
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Относительный путь

# Импортируем функционал PyTorch для работы с нейронными сетями
import torch.nn.functional as F

# Импортируем необходимые модули и функции из различных утилит и моделей YOLOv5
from models.common import DetectMultiBackend
from models.yolo import SegmentationModel
from utils.callbacks import Callbacks
from utils.general import (LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size,
                           check_requirements, check_yaml, coco80_to_coco91_class, colorstr, increment_path,
                           non_max_suppression, print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, box_iou
from utils.plots import output_to_target, plot_val_study
from utils.segment.dataloaders import create_dataloader
from utils.segment.general import mask_iou, process_mask, process_mask_upsample, scale_image
from utils.segment.metrics import Metrics, ap_per_class_box_and_mask
from utils.segment.plots import plot_images_and_masks
from utils.torch_utils import de_parallel, select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    # Сохраняем один результат в текстовый файл
    # Вычисляем коэффициенты нормализации для преобразования координат боксов
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # Коэффициенты нормализации whwh
    # Проходим по каждому предсказанию
    for *xyxy, conf, cls in predn.tolist():
        # Преобразуем координаты боксов из формата xyxy в xywh и нормализуем
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # Нормализованные xywh
        # Формируем строку для записи в файл
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # Формат метки
        # Открываем файл для добавления данных и записываем строку
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map, pred_masks):
    # Сохраняем один результат в формате JSON
    # {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    from pycocotools.mask import encode

    def single_encode(x):
        # Кодируем маску в формат RLE
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    # Получаем идентификатор изображения из имени файла
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    # Преобразуем координаты боксов из формата xyxy в xywh
    box = xyxy2xywh(predn[:, :4])  # xywh
    # Перемещаем центр бокса в верхний левый угол
    box[:, :2] -= box[:, 2:] / 2  # Центр xy в верхний левый угол
    # Транспонируем маски для правильного порядка размерностей
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    # Используем пул потоков для параллельной обработки масок
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    # Проходим по каждому предсказанию и добавляем информацию в JSON
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5),
            'segmentation': rles[i]
        })


def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    """
    Возвращает матрицу правильных предсказаний
    Аргументы:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Возвращает:
        correct (array[N, 10]), для 10 уровней IoU
    """
    if masks:
        if overlap:
            # Получаем количество меток
            nl = len(labels)
            # Создаем индекс для маркировки каждой маски
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            # Повторяем маски истинных значений для каждой метки
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
            # Устанавливаем значения маски в 1 только там, где индекс совпадает
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        # Если размеры маски истинных значений и предсказанных маск не совпадают, интерполируем
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        # Вычисляем IoU между масками истинных значений и предсказанных маск
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
    else:  # Работаем с боксами
        # Вычисляем IoU между боксами истинных значений и предсказанных боксов
        iou = box_iou(labels[:, 1:], detections[:, :4])

    # Инициализируем матрицу правильных предсказаний
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    # Проверяем соответствие классов между метками и предсказаниями
    correct_class = labels[:, 0:1] == detections[:, 5]
    # Проходим по каждому уровню IoU
    for i in range(len(iouv)):
        # Находим индексы, где IoU выше порога и классы совпадают
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > порог и классы совпадают
        if x[0].shape[0]:
            # Создаем матрицу совпадений [метка, предсказание, IoU]
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                # Сортируем совпадения по убыванию IoU
                matches = matches[matches[:, 2].argsort()[::-1]]
                # Удаляем дубликаты по индексу предсказания
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # Снова сортируем по убыванию IoU
                # matches = matches[matches[:, 2].argsort()[::-1]]
                # Удаляем дубликаты по индексу метки
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            # Устанавливаем правильные предсказания в матрице
            correct[matches[:, 1].astype(int), i] = True
    # Возвращаем матрицу правильных предсказаний в виде тензора
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

@smart_inference_mode()
def run(
        data,
        weights=None,  # Путь(и) к файлу(ам) модели model.pt
        batch_size=32,  # Размер пакета
        imgsz=640,  # Размер изображения для вывода (в пикселях)
        conf_thres=0.001,  # Порог уверенности
        iou_thres=0.6,  # Порог IoU для NMS
        max_det=300,  # Максимальное количество обнаружений на изображение
        task='val',  # Задача: train, val, test, speed или study
        device='',  # Устройство CUDA, например 0 или 0,1,2,3 или cpu
        workers=8,  # Максимальное количество рабочих процессов загрузчика данных (на RANK в режиме DDP)
        single_cls=False,  # Рассматривать как одноклассовый набор данных
        augment=False,  # Усиленное вывод
        verbose=False,  # Подробный вывод
        save_txt=False,  # Сохранять результаты в файл *.txt
        save_hybrid=False,  # Сохранять гибридные результаты (метка + предсказание) в файл *.txt
        save_conf=False,  # Сохранять уверенности в метках --save-txt
        save_json=False,  # Сохранять файл результатов в формате COCO - JSON
        project=ROOT / 'runs/val-seg',  # Сохранять в project/name
        name='exp',  # Сохранять в project/name
        exist_ok=False,  # Если существующий project/name ок, не увеличивать номер
        half=True,  # Использовать полноразмерную точность FP16 для вывода
        dnn=False,  # Использовать OpenCV DNN для вывода ONNX
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        overlap=False,
        mask_downsample_ratio=1,
        compute_loss=None,
        callbacks=Callbacks(),
):
    if save_json:
        # Если нужно сохранить в JSON, убеждаемся, что есть необходимые библиотеки
        check_requirements(['pycocotools'])
        # Выбираем более точный метод обработки маски
        process = process_mask_upsample
    else:
        # В противном случае выбираем более быстрый метод обработки маски
        process = process_mask

    # Инициализация/загрузка модели и настройка устройства
    training = model is not None
    if training:  # Вызывается из train.py
        # Получаем устройство модели и флаги о типе модели
        device, pt, jit, engine = next(model.parameters()).device, True, False, False
        # Полноразмерная точность поддерживается только на CUDA
        half &= device.type != 'cpu'
        # Устанавливаем точность модели
        model.half() if half else model.float()
        # Получаем количество масок
        nm = de_parallel(model).model[-1].nm
    else:  # Вызывается напрямую
        # Выбираем устройство
        device = select_device(device, batch_size=batch_size)

        # Создание директорий
        # Увеличиваем номер запуска, если необходимо
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        # Создаем директорию для сохранения результатов
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

        # Загрузка модели
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        # Получаем параметры модели
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        # Проверяем размер изображения
        imgsz = check_img_size(imgsz, s=stride)
        # Получаем информацию о поддержке FP16
        half = model.fp16
        # Получаем количество масок
        nm = de_parallel(model).model.model[-1].nm if isinstance(model, SegmentationModel) else 32
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # Модели из export.py по умолчанию имеют размер пакета 1
                LOGGER.info(f'Принудительно устанавливаем --batch-size 1 для квадратного вывода (1,3,{imgsz},{imgsz}) для не-PyTorch моделей')

        # Проверка данных
        data = check_dataset(data)

    # Конфигурация
    model.eval()
    cuda = device.type != 'cpu'
    # Проверка, является ли набор данных COCO
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')
    # Определяем количество классов
    nc = 1 if single_cls else int(data['nc'])
    # Создаем вектор IoU для mAP@0.5:0.95
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    # Загрузчик данных
    if not training:
        if pt and not single_cls:  # Проверяем, что веса модели соответствуют данным
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} классов) обучена на других --data, чем те, которые вы передали ({nc} классов). Передайте правильную комбинацию --weights и --data, которые были обучены вместе.'
        # Разогрев модели
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))
        # Параметры для квадратного вывода в бенчмарках
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)
        # Определяем задачу
        task = task if task in ('train', 'val', 'test') else 'val'
        # Создаем загрузчик данных
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '),
                                       overlap_mask=overlap,
                                       mask_downsample_ratio=mask_downsample_ratio)[0]

    seen = 0
    # Создаем матрицу ошибок
    confusion_matrix = ConfusionMatrix(nc=nc)
    # Получаем имена классов
    names = model.names if hasattr(model, 'names') else model.module.names
    if isinstance(names, (list, tuple)):  # Старый формат
        names = dict(enumerate(names))
    # Создаем карту классов для COCO
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # Форматируем строку заголовка для вывода статистики
    s = ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95)", "Mask(P", "R",
                                  "mAP50", "mAP50-95)")
    # Создаем профилировщики времени
    dt = Profile(), Profile(), Profile()
    # Создаем объект для хранения метрик
    metrics = Metrics()
    # Инициализируем тензор потерь
    loss = torch.zeros(4, device=device)
    # Список для хранения данных в формате JSON
    jdict, stats = [], []
    # callbacks.run('on_val_start')
    # Создаем прогресс - бар
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)
    for batch_i, (im, targets, paths, shapes, masks) in enumerate(pbar):
        # callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                # Переносим изображения на устройство CUDA
                im = im.to(device, non_blocking=True)
                # Переносим метки на устройство CUDA
                targets = targets.to(device)
                # Переносим маски на устройство CUDA
                masks = masks.to(device)
            masks = masks.float()
            # Преобразуем изображения в нужную точность
            im = im.half() if half else im.float()
            # Нормализуем изображения
            im /= 255
            # Получаем размеры пакета, каналов, высоты и ширины изображения
            nb, _, height, width = im.shape

        # Инференс (вычисление предсказаний)
        with dt[1]:
            # Если указана функция для вычисления потерь, то получаем предсказания, прото - маски и выход тренировочной модели
            # В противном случае получаем только предсказания и прото - маски
            preds, protos, train_out = model(im) if compute_loss else (*model(im, augment=augment)[:2], None)

        # Вычисление потерь
        if compute_loss:
            # Увеличиваем суммарные потери на потери текущего пакета (бокс, объект, класс)
            loss += compute_loss((train_out, protos), targets, masks)[1]

            # Нелокальное подавление максимумов (NMS)
        # Преобразуем координаты меток из нормализованных значений в пиксели
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        # Создаем список меток для каждого изображения в пакете, если нужно сохранять гибридные результаты
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        with dt[2]:
            # Применяем NMS к предсказаниям
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det,
                                        nm=nm)

        # Вычисление метрик
        plot_masks = []  # Маски для визуализации
        for si, (pred, proto) in enumerate(zip(preds, protos)):
            # Получаем метки для текущего изображения
            labels = targets[targets[:, 0] == si, 1:]
            # Количество меток и предсказаний
            nl, npr = labels.shape[0], pred.shape[0]
            # Путь к текущему изображению и его исходная форма
            path, shape = Path(paths[si]), shapes[si][0]
            # Инициализируем матрицы правильных предсказаний для маск и боксов
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            # Увеличиваем счетчик просмотренных изображений
            seen += 1

            if npr == 0:
                if nl:
                    # Добавляем статистику для пустых предсказаний, если есть метки
                    stats.append((correct_masks, correct_bboxes, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        # Обрабатываем пакет для матрицы ошибок
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Обработка масок
            midx = [si] if overlap else targets[:, 0] == si
            # Получаем маски истинных значений для текущего изображения
            gt_masks = masks[midx]
            # Получаем предсказанные маски
            pred_masks = process(proto, pred[:, 6:], pred[:, :4], shape=im[si].shape[1:])

            # Предсказания
            if single_cls:
                # Если рассматриваем как одноклассовый набор данных, устанавливаем класс всех предсказаний равным 0
                pred[:, 5] = 0
            predn = pred.clone()
            # Масштабируем координаты боксов до исходного размера изображения
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            # Оценка
            if nl:
                # Преобразуем координаты меток из формата xywh в xyxy
                tbox = xywh2xyxy(labels[:, 1:5])
                # Масштабируем координаты меток до исходного размера изображения
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                # Объединяем метки класса и координаты боксов в одной тензоре
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                # Вычисляем правильные предсказания для боксов
                correct_bboxes = process_batch(predn, labelsn, iouv)
                # Вычисляем правильные предсказания для масок
                correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap=overlap, masks=True)
                if plots:
                    # Обрабатываем пакет для матрицы ошибок
                    confusion_matrix.process_batch(predn, labelsn)
            # Добавляем статистику для текущего изображения
            stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))

            # Преобразуем предсказанные маски в тип uint8
            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if plots and batch_i < 3:
                # Фильтруем топ - 15 предсказанных масок для визуализации
                plot_masks.append(pred_masks[:15].cpu())

                # Сохранение/логирование
            if save_txt:
                # Сохраняем предсказания в текстовый файл
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                # Масштабируем предсказанные маски до исходного размера изображения
                pred_masks = scale_image(im[si].shape[1:],
                                         pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[si][1])
                # Добавляем предсказания в словарь для сохранения в формате COCO - JSON
                save_one_json(predn, jdict, path, class_map, pred_masks)
                # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Визуализация изображений
        if plots and batch_i < 3:
            if len(plot_masks):
                # Объединяем все маски для визуализации в один тензор
                plot_masks = torch.cat(plot_masks, dim=0)
            # Визуализируем изображения с метками
            plot_images_and_masks(im, targets, masks, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)
            # Визуализируем изображения с предсказаниями
            plot_images_and_masks(im, output_to_target(preds, max_det=15), plot_masks, paths,
                                  save_dir / f'val_batch{batch_i}_pred.jpg', names)
            # callbacks.run('on_val_batch_end')

        # Вычисление метрик
        # Преобразуем статистику в numpy массивы
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        if len(stats) and stats[0].any():
            # Вычисляем метрики для боксов и масок
            results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names)
            # Обновляем метрики
            metrics.update(results)
        # Количество целей для каждого класса
        nt = np.bincount(stats[4].astype(int), minlength=nc)

        # Вывод результатов
        # Формат вывода результатов
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 8
        # Выводим средние метрики для всех классов
        LOGGER.info(pf % ("all", seen, nt.sum(), *metrics.mean_results()))
        if nt.sum() == 0:
            # Предупреждение, если в наборе данных нет меток
            LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

        # Вывод результатов для каждого класса
        if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
            for i, c in enumerate(metrics.ap_class_index):
                # Выводим метрики для каждого класса
                LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i)))

        # Вывод скорости обработки
        # Скорость обработки на одно изображение
        t = tuple(x.t / seen * 1E3 for x in dt)
        if not training:
            shape = (batch_size, 3, imgsz, imgsz)
            # Выводим время пред - обработки, инференса и NMS на одно изображение
            LOGGER.info(f'Speed: %.1fms pre - process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Визуализация (Построение графиков)
    if plots:
        # Построение и сохранение графика матрицы ошибок
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    # callbacks.run('on_val_end')

    # Получаем средние результаты метрик для боксов и масок
    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = metrics.mean_results()

    # Сохранение в JSON
    if save_json and len(jdict):
        # Получаем имя файла с весами модели
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        # Путь к файлу с аннотациями COCO
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')
        # Путь к файлу с предсказаниями в формате JSON
        pred_json = str(save_dir / f"{w}_predictions.json")
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            # Сохраняем словарь с предсказаниями в файл JSON
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # Инициализация API для работы с аннотациями
            anno = COCO(anno_json)
            # Инициализация API для работы с предсказаниями
            pred = anno.loadRes(pred_json)
            results = []
            for eval in COCOeval(anno, pred, 'bbox'), COCOeval(anno, pred, 'segm'):
                if is_coco:
                    # Устанавливаем идентификаторы изображений для оценки
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]
                    # Выполняем оценку
                eval.evaluate()
                # Аккумулируем результаты
                eval.accumulate()
                # Сводим результаты
                eval.summarize()
                # Добавляем первые два значения статистики в результаты
                results.extend(eval.stats[:2])
                # Обновляем значения метрик для боксов и масок
            map_bbox, map50_bbox, map_mask, map50_mask = results
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Возврат результатов
    # Возвращаем модель в формат с полной точностью (для обучения)
    model.float()
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # Собираем финальные метрики
    final_metric = mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask
    # Возвращаем финальные метрики, карты метрик для каждого класса и время обработки
    return (*final_metric, *(loss.cpu() / len(dataloader)).tolist()), metrics.get_maps(nc), t

    def parse_opt():
        # Создаем парсер аргументов командной строки
        parser = argparse.ArgumentParser()
        # Аргумент для пути к файлу с данными о наборе данных
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128-seg.yaml', help='путь к dataset.yaml')
        # Аргумент для пути(ей) к файлу(ам) модели
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt',
                            help='путь(и) к модели(ям)')
        # Аргумент для размера пакета
        parser.add_argument('--batch-size', type=int, default=32, help='размер пакета')
        # Аргумент для размера изображения для вывода
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640,
                            help='размер изображения для вывода (в пикселях)')
        # Аргумент для порога уверенности
        parser.add_argument('--conf-thres', type=float, default=0.001, help='порог уверенности')
        # Аргумент для порога IoU для NMS
        parser.add_argument('--iou-thres', type=float, default=0.6, help='порог IoU для NMS')
        # Аргумент для максимального количества обнаружений на изображение
        parser.add_argument('--max-det', type=int, default=300,
                            help='максимальное количество обнаружений на изображение')
        # Аргумент для типа задачи
        parser.add_argument('--task', default='val', help='train, val, test, speed или study')
        # Аргумент для устройства CUDA
        parser.add_argument('--device', default='', help='устройство CUDA, например 0 или 0,1,2,3 или cpu')
        # Аргумент для максимального количества рабочих процессов загрузчика данных
        parser.add_argument('--workers', type=int, default=8,
                            help='максимальное количество рабочих процессов загрузчика данных (на RANK в режиме DDP)')
        # Аргумент для обработки как одноклассового набора данных
        parser.add_argument('--single-cls', action='store_true', help='рассматривать как одноклассовый набор данных')
        # Аргумент для усиленного вывода
        parser.add_argument('--augment', action='store_true', help='усиленное вывод')
        # Аргумент для подробного вывода метрик по классам
        parser.add_argument('--verbose', action='store_true', help='отчёт о mAP по классам')
        # Аргумент для сохранения результатов в текстовый файл
        parser.add_argument('--save-txt', action='store_true', help='сохранять результаты в *.txt')
        # Аргумент для сохранения гибридных результатов (метка + предсказание) в текстовый файл
        parser.add_argument('--save-hybrid', action='store_true',
                            help='сохранять гибридные результаты (метка + предсказание) в *.txt')
        # Аргумент для сохранения уверенностей в метках при сохранении в текстовый файл
        parser.add_argument('--save-conf', action='store_true', help='сохранять уверенности в метках --save-txt')
        # Аргумент для сохранения файла результатов в формате COCO - JSON
        parser.add_argument('--save-json', action='store_true', help='сохранять файл результатов в формате COCO - JSON')
        # Аргумент для пути к проекту для сохранения результатов
        parser.add_argument('--project', default=ROOT / 'runs/val-seg', help='сохранять результаты в project/name')
        # Аргумент для имени проекта для сохранения результатов
        parser.add_argument('--name', default='exp', help='сохранять в project/name')
        # Аргумент для разрешения использования существующего проекта/имени без увеличения номера
        parser.add_argument('--exist-ok', action='store_true',
                            help='существующий project/name ок, не увеличивать номер')
        # Аргумент для использования полноразмерной точности FP16 для вывода
        parser.add_argument('--half', action='store_true', help='использовать полноразмерную точность FP16 для вывода')
        # Аргумент для использования OpenCV DNN для вывода ONNX
        parser.add_argument('--dnn', action='store_true', help='использовать OpenCV DNN для вывода ONNX')
        # Парсим аргументы командной строки
        opt = parser.parse_args()
        # Проверяем правильность пути к файлу с данными (YAML)
        opt.data = check_yaml(opt.data)
        # opt.save_json |= opt.data.endswith('coco.yaml')
        # Если нужно сохранять гибридные результаты, то также сохраняем в текстовый файл
        opt.save_txt |= opt.save_hybrid
        # Выводим аргументы командной строки
        print_args(vars(opt))
        # Возвращаем объект с аргументами
        return opt

def main(opt):
    # Проверка требований по зависимостям, исключая tensorboard и thop
    check_requirements(requirements=ROOT /'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # Обычный запуск
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            # Вывод предупреждения о некорректном пороге уверенности
            LOGGER.warning(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            # Вывод предупреждения о сохранении гибридных результатов
            LOGGER.warning('WARNING ⚠️ --save-hybrid returns high mAP from hybrid labels, not from predictions alone')
        # Запуск функции run с аргументами из объекта opt
        run(**vars(opt))

    else:
        # Преобразуем weights в список, если это не список уже
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        # Определяем, можно ли использовать FP16 на основе доступности CUDA и выбранного устройства
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'
        if opt.task =='speed':  # бенчмарки скорости
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            # Устанавливаем значения порогов уверенности и IoU, а также отключаем сохранение в JSON
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                # Запуск функции run с аргументами из объекта opt, без построения графиков
                run(**vars(opt), plots=False)

        elif opt.task =='study':  # бенчмарки скорости против mAP
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                # Создаем имя файла для сохранения результатов
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'
                x, y = list(range(256, 1536 + 128, 128)), []  # ось x (размеры изображений), ось y
                for opt.imgsz in x:  # размер изображения
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    # Запуск функции run с аргументами из объекта opt, без построения графиков
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # результаты и времена
                # Сохраняем результаты в файл
                np.savetxt(f, y, fmt='%10.4g')
            # Создаем архив с файлами результатов
            os.system('zip -r study.zip study_*.txt')
            # Построение графика исследования
            plot_val_study(x=x)


if __name__ == "__main__":
    # Парсим аргументы командной строки
    opt = parse_opt()
    # Запускаем основную функцию
    main(opt)