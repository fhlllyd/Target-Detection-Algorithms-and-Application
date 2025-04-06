# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Модуль для загрузки данных и утилит для работы с данными
"""

import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

# ... (Импорты остальных библиотек)

# Параметры
HELP_URL = 'См. https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'  # Ссылка на справку
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # Поддерживаемые форматы изображений
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # Поддерживаемые форматы видео
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # Локальный ранг для分布式 обучения
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # Глобальный флаг для закрепления памяти

# Получение тега ориентации EXIF
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def get_hash(paths):
    # Возвращает хеш-сумму списка путей (файлов или директорий)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # Размеры файлов
    h = hashlib.md5(str(size).encode())  # Хеш размеров
    h.update(''.join(paths).encode())  # Хеш путей
    return h.hexdigest()  # Возвращает хеш-сумму

def exif_size(img):
    # Возвращает размер PIL с учетом EXIF-ориентации
    s = img.size  # (ширина, высота)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # Поворот на 270 или 90 градусов
            s = (s[1], s[0])
    except:
        pass
    return s

# ... (Остальные функции и классы)

class LoadImagesAndLabels(Dataset):
    # Класс для загрузки изображений и меток для обучения и валидации YOLOv5
    cache_version = 0.6  # Версия кэша меток
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
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
                 pad=0.0,
                 min_items=0,
                 prefix=''):
        # Инициализация параметров
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = rect
        self.mosaic = augment and not rect  # Использовать мозаику (только при аугментации)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None

        # Загрузка списка изображений
        try:
            # Поиск изображений в директории или файле
            f = []  # Список путей к изображениям
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():  # Если путь — директория
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # Если путь — файл (например, текстовый файл с путями)
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]
            self.im_files = sorted(x for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            assert self.im_files, f'{prefix}Не найдено изображений'
        except Exception as e:
            raise Exception(f'{prefix}Ошибка загрузки данных из {path}: {e}\n{HELP_URL}') from e

        # Загрузка меток и проверка кэша
        self.label_files = img2label_paths(self.im_files)  # Пути к меткам
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True
            assert cache['version'] == self.cache_version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False

        # ... (Остальной код класса)

    def __getitem__(self, index):
        # Загрузка и аугментация изображения
        index = self.indices[index]  # Линейный, перемешанный или взвешенный индекс

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Загрузка мозаики (4 изображения)
            img, labels = self.load_mosaic(index)
            shapes = None

            # Применение MixUp
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Загрузка обычного изображения
            img, (h0, w0), (h, w) = self.load_image(index)

            # Пропорциональное изменение размера (letterbox)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        # ... (Остальной код метода)

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
                      shuffle=False):
    # Создание даталоадера для обучения или валидации
    if rect and shuffle:
        LOGGER.warning('ПРЕДУПРЕЖДЕНИЕ ⚠️ --rect несовместим сshuffle, shuffle выключен')
        shuffle = False
    with torch_distributed_zero_first(rank):  # Инициализация кэша только один раз при分布式 обучении
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset