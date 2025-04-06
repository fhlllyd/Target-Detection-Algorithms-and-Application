# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, distributed

from ..augmentations import augment_hsv, copy_paste, letterbox
from ..dataloaders import InfiniteDataLoader, LoadImagesAndLabels, seed_worker
from ..general import LOGGER, xyn2xy, xywhn2xyxy, xyxy2xywhn
from ..torch_utils import torch_distributed_zero_first
from .augmentations import mixup, random_perspective

# Получаем значение переменной окружения 'RANK'. Если переменная не определена, то используем значение -1
RANK = int(os.getenv('RANK', -1))

def create_dataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False,
                      mask_downsample_ratio=1,
                      overlap_mask=False):
    # Проверяем, если включены оба параметра rect и shuffle
    if rect and shuffle:
        # Выводим предупреждение, что параметры rect и shuffle несовместимы, и устанавливаем shuffle в False
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    # Инициализируем кэш датасета только один раз, если используется распределенное обучение (DDP)
    with torch_distributed_zero_first(rank):
        # Создаем экземпляр класса LoadImagesAndLabelsAndMasks для загрузки изображений, меток и масок
        dataset = LoadImagesAndLabelsAndMasks(
            path,
            imgsz,
            batch_size,
            augment=augment,  # Аугментация
            hyp=hyp,  # Гиперпараметры
            rect=rect,  # Прямоугольные батчи
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            downsample_ratio=mask_downsample_ratio,
            overlap=overlap_mask)

    # Убеждаемся, что размер батча не превышает количество элементов в датасете
    batch_size = min(batch_size, len(dataset))
    # Получаем количество доступных CUDA-устройств
    nd = torch.cuda.device_count()
    # Вычисляем количество рабочих процессов для загрузки данных
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    # Создаем семплер для распределенного обучения, если ранг не равен -1
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    # Выбираем тип загрузчика данных в зависимости от использования весов изображений
    loader = DataLoader if image_weights else InfiniteDataLoader
    # Создаем генератор случайных чисел и устанавливаем начальное значение
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    # Возвращаем загрузчик данных и датасет
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabelsAndMasks.collate_fn4 if quad else LoadImagesAndLabelsAndMasks.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


class LoadImagesAndLabelsAndMasks(LoadImagesAndLabels):  # Для обучения/тестирования

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0,
        min_items=0,
        prefix="",
        downsample_ratio=1,
        overlap=False,
    ):
        # Вызываем конструктор родительского класса
        super().__init__(path, img_size, batch_size, augment, hyp, rect, image_weights, cache_images, single_cls,
                         stride, pad, min_items, prefix)
        # Устанавливаем коэффициент уменьшения масштаба маски
        self.downsample_ratio = downsample_ratio
        # Устанавливаем флаг перекрытия масок
        self.overlap = overlap

    def __getitem__(self, index):
        # Получаем индекс элемента в соответствии с выбранной стратегией (линейная, перемешанная или по весам изображений)
        index = self.indices[index]

        # Получаем гиперпараметры
        hyp = self.hyp
        # Проверяем, нужно ли использовать мозаику и генерируем случайное число для принятия решения
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        # Инициализируем пустой список для хранения масок
        masks = []
        if mosaic:
            # Загружаем изображение в формате мозаики
            img, labels, segments = self.load_mosaic(index)
            # Инициализируем переменную для хранения информации о форме изображения
            shapes = None

            # Проверяем, нужно ли применить аугментацию MixUp и генерируем случайное число для принятия решения
            if random.random() < hyp["mixup"]:
                # Применяем аугментацию MixUp
                img, labels, segments = mixup(img, labels, segments, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Загружаем отдельное изображение
            img, (h0, w0), (h, w) = self.load_image(index)

            # Применяем letterbox преобразование к изображению
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            # Сохраняем информацию о форме изображения до и после преобразования
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            # Копируем метки для текущего изображения
            labels = self.labels[index].copy()
            # Копируем сегменты для текущего изображения
            segments = self.segments[index].copy()
            if len(segments):
                for i_s in range(len(segments)):
                    # Преобразуем нормализованные координаты сегментов в пиксельные координаты
                    segments[i_s] = xyn2xy(
                        segments[i_s],
                        ratio[0] * w,
                        ratio[1] * h,
                        padw=pad[0],
                        padh=pad[1],
                    )
            if labels.size:
                # Преобразуем нормализованные координаты bounding box-ов в пиксельные координаты
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                # Применяем случайную перспективную аугментацию к изображению, меткам и сегментам
                img, labels, segments = random_perspective(img,
                                                           labels,
                                                           segments=segments,
                                                           degrees=hyp["degrees"],
                                                           translate=hyp["translate"],
                                                           scale=hyp["scale"],
                                                           shear=hyp["shear"],
                                                           perspective=hyp["perspective"])

        # nl = количество меток
        nl = len(labels)
        # Проверка, есть ли метки
        if nl:
            # Преобразование координат из формата xyxy в xywhn и ограничение значений
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            # Если включено перекрытие
            if self.overlap:
                # Преобразование полигонов в маски с учетом перекрытия
                masks, sorted_idx = polygons2masks_overlap(img.shape[:2],
                                                           segments,
                                                           downsample_ratio=self.downsample_ratio)
                # Добавление дополнительного измерения к маскам
                masks = masks[None]  # (640, 640) -> (1, 640, 640)
                # Сортировка меток в соответствии с отсортированными индексами
                labels = labels[sorted_idx]
            else:
                # Преобразование полигонов в маски без учета перекрытия
                masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=self.downsample_ratio)

        # Преобразование массива маск в тензор PyTorch или создание пустого тензора
        masks = (torch.from_numpy(masks) if len(masks) else torch.zeros(1 if self.overlap else nl, img.shape[0] //
                                                                        self.downsample_ratio, img.shape[1] //
                                                                        self.downsample_ratio))
        # TODO: Поддержка библиотеки albumentations
        if self.augment:
            # Albumentations
            # Некоторые аугментации не влияют на боксы и маски, поэтому пока оставляем так.
            img, labels = self.albumentations(img, labels)
            # Обновление количества меток после аугментации
            nl = len(labels)

            # Изменение цветового пространства HSV
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Переворот изображения по вертикали
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    # Обновление координат меток после переворота
                    labels[:, 2] = 1 - labels[:, 2]
                    # Переворот маск по вертикали
                    masks = torch.flip(masks, dims=[1])

            # Переворот изображения по горизонтали
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    # Обновление координат меток после переворота
                    labels[:, 1] = 1 - labels[:, 1]
                    # Переворот маск по горизонтали
                    masks = torch.flip(masks, dims=[2])

            # Cutouts  # labels = cutout(img, labels, p=0.5)

        # Создание тензора для выходных меток
        labels_out = torch.zeros((nl, 6))
        if nl:
            # Копирование меток в выходной тензор
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Преобразование формата изображения из HWC в CHW и из BGR в RGB
        img = img.transpose((2, 0, 1))[::-1]
        # Создание непрерывного массива
        img = np.ascontiguousarray(img)

        # Возвращение тензора изображения, тензора меток, пути к файлу, формата и тензора масок
        return (torch.from_numpy(img), labels_out, self.im_files[index], shapes, masks)

        def load_mosaic(self, index):
            # Загрузчик 4-мозаики в YOLOv5. Загружает 1 изображение + 3 случайных изображения в 4-изображение мозаику
            labels4, segments4 = [], []
            # Размер изображения
            s = self.img_size
            # Координаты центра мозаики
            yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)

            # 3 дополнительных индекса изображений
            indices = [index] + random.choices(self.indices, k=3)
            for i, index in enumerate(indices):
                # Загрузка изображения
                img, _, (h, w) = self.load_image(index)

                # Размещение изображения в img4
                if i == 0:  # Верхний левый угол
                    img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                elif i == 1:  # Верхний правый угол
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # Нижний левый угол
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # Нижний правый угол
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                # Копирование изображения в img4
                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
                # Вычисление отступов
                padw = x1a - x1b
                padh = y1a - y1b

                # Копирование меток и сегментов
                labels, segments = self.labels[index].copy(), self.segments[index].copy()

                if labels.size:
                    # Преобразование нормализованных координат xywh в пиксельные координаты xyxy
                    labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
                    # Преобразование нормализованных координат сегментов в пиксельные координаты
                    segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
                # Добавление меток и сегментов в списки
                labels4.append(labels)
                segments4.extend(segments)

            # Объединение меток и ограничение значений
            labels4 = np.concatenate(labels4, 0)
            for x in (labels4[:, 1:], *segments4):
                np.clip(x, 0, 2 * s, out=x)  # Ограничение значений при использовании random_perspective()
            # img4, labels4 = replicate(img4, labels4)  # Репликация

            # Аугментация
            img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
            img4, labels4, segments4 = random_perspective(img4,
                                                          labels4,
                                                          segments4,
                                                          degrees=self.hyp["degrees"],
                                                          translate=self.hyp["translate"],
                                                          scale=self.hyp["scale"],
                                                          shear=self.hyp["shear"],
                                                          perspective=self.hyp["perspective"],
                                                          border=self.mosaic_border)  # Граница для удаления
            return img4, labels4, segments4

        @staticmethod
        def collate_fn(batch):
            # Распаковка батча
            img, label, path, shapes, masks = zip(*batch)
            # Объединение масок в один тензор
            batched_masks = torch.cat(masks, 0)
            for i, l in enumerate(label):
                # Добавление индекса целевого изображения для build_targets()
                l[:, 0] = i
            return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks

        def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
            """
            Аргументы:
                img_size (tuple): Размер изображения.
                polygons (np.ndarray): [N, M], где N - количество полигонов,
                    M - количество точек (делится на 2).
            """
            # Создание пустой маски
            mask = np.zeros(img_size, dtype=np.uint8)
            # Преобразование полигонов в массив numpy
            polygons = np.asarray(polygons)
            # Преобразование типа данных полигонов в int32
            polygons = polygons.astype(np.int32)
            # Получение формы полигонов
            shape = polygons.shape
            # Изменение формы полигонов
            polygons = polygons.reshape(shape[0], -1, 2)
            # Заполнение полигонов на маске
            cv2.fillPoly(mask, polygons, color=color)
            # Вычисление новых размеров маски
            nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
            # Сжатие маски
            mask = cv2.resize(mask, (nw, nh))
            return mask

        def polygons2masks(img_size, polygons, color, downsample_ratio=1):
            """
            Аргументы:
                img_size (tuple): Размер изображения.
                polygons (list[np.ndarray]): каждый полигон имеет форму [N, M],
                    N - количество полигонов,
                    M - количество точек (делится на 2).
            """
            masks = []
            for si in range(len(polygons)):
                # Создание маски для каждого полигона
                mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
                masks.append(mask)
            return np.array(masks)

        def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
            """Возвращает маску перекрытия размером (640, 640)."""
            # Создание пустой маски перекрытия
            masks = np.zeros((img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
                             dtype=np.int32 if len(segments) > 255 else np.uint8)
            areas = []
            ms = []
            for si in range(len(segments)):
                # Создание маски для каждого сегмента
                mask = polygon2mask(
                    img_size,
                    [segments[si].reshape(-1)],
                    downsample_ratio=downsample_ratio,
                    color=1,
                )
                ms.append(mask)
                # Вычисление площади маски
                areas.append(mask.sum())
            # Преобразование списка площадей в массив numpy
            areas = np.asarray(areas)
            # Сортировка индексов по убыванию площадей
            index = np.argsort(-areas)
            # Сортировка масок по убыванию площадей
            ms = np.array(ms)[index]
            for i in range(len(segments)):
                # Умножение маски на номер сегмента
                mask = ms[i] * (i + 1)
                # Добавление маски к маске перекрытия
                masks = masks + mask
                # Ограничение значений маски перекрытия
                masks = np.clip(masks, a_min=0, a_max=i + 1)
            return masks, index