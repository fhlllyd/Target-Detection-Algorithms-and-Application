# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy
from utils.metrics import bbox_ioa

# Средние значения по каналам RGB для ImageNet
IMAGENET_MEAN = 0.485, 0.456, 0.406
# Стандартные отклонения по каналам RGB для ImageNet
IMAGENET_STD = 0.229, 0.224, 0.225


class Albumentations:
    # Класс Albumentations для YOLOv5 (необязательно, используется только если пакет установлен)
    def __init__(self, size=640):
        self.transform = None
        # Префикс для логгирования
        prefix = colorstr('albumentations: ')
        try:
            import albumentations as A
            # Проверка версии Albumentations
            check_version(A.__version__, '1.0.3', hard=True)

            # Список аугментаций
            T = [
                # Случайное изменение размера и обрезка
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                # Размытие
                A.Blur(p=0.01),
                # Медианное размытие
                A.MedianBlur(p=0.01),
                # Преобразование в оттенки серого
                A.ToGray(p=0.01),
                # Контрастное ограничение адаптивной гистограммы
                A.CLAHE(p=0.01),
                # Случайное изменение яркости и контраста
                A.RandomBrightnessContrast(p=0.0),
                # Случайное изменение гаммы
                A.RandomGamma(p=0.0),
                # Сжатие изображения
                A.ImageCompression(quality_lower=75, p=0.0)
            ]
            # Создание композитного преобразования
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            # Логгирование активных аугментаций
            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        except ImportError:
            # Пропустить, если пакет не установлен
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    def __call__(self, im, labels, p=1.0):
        # Применить преобразования, если они определены и случайное число меньше p
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            im = new['image']
            labels = np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    # Нормализация RGB-изображений x по статистикам ImageNet в формате BCHW
    return TF.normalize(x, mean, std, inplace=inplace)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # Де-нормализация RGB-изображений x по статистикам ImageNet в формате BCHW
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # Аугментация в цветовом пространстве HSV
    if hgain or sgain or vgain:
        # Случайные коэффициенты изменения
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        # Преобразование в цветовое пространство HSV
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        # Таблицы преобразования для каждого канала
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # Применение таблиц преобразования и обратное преобразование в BGR
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)


def hist_equalize(im, clahe=True, bgr=False):
    # Выравнивание гистограммы на BGR-изображении 'im'
    # Преобразование в цветовое пространство YUV
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        # Контрастное ограничение адаптивной гистограммы
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        # Выравнивание гистограммы по каналу Y
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    # Обратное преобразование в RGB или BGR
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)


def replicate(im, labels):
    # Репликация меток на изображении
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    # Средняя сторона bounding box
    s = ((x2 - x1) + (y2 - y1)) / 2
    for i in s.argsort()[:round(s.size * 0.5)]:
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        # Случайное смещение
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        # Копирование области изображения
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]
        # Добавление новой метки
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Изменение размера и заполнение изображения с соблюдением ограничений шага
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Коэффициент масштабирования
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    # Новый размер без заполнения
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # Величина заполнения по ширине и высоте
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        # Минимальный прямоугольник
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        # Растягивание
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        # Изменение размера изображения
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # Добавление границ
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def random_perspective(im,
                       targets=(),
                       segments=(),
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # Случайная афинная и перспективная трансформация изображения
    height = im.shape[0] + border[0] * 2
    width = im.shape[1] + border[1] * 2

    # Матрица сдвига в центр
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2
    C[1, 2] = -im.shape[0] / 2

    # Матрица перспективы
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    # Матрица вращения и масштабирования
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Матрица сдвига
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # Матрица трансляции
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Общая матрица преобразования
    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            # Перспективное преобразование
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            # Аффинное преобразование
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Преобразование координат меток
        n = len(targets)
        if n:
            use_segments = any(x.any() for x in segments)
            new = np.zeros((n, 4))
            if use_segments:  # А핀ное преобразование сегментов
                segments = resample_segments(segments)  # Увеличение取样率
                for i, segment in enumerate(segments):
                    xy = np.ones((len(segment), 3))
                    xy[:, :2] = segment
                    xy = xy @ M.T  # Трансформация
                    xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:,
                                                                    :2]  # Пресеклирование перспективы или аффинное преобразование

                    # Обрезка
                    new[i] = segment2box(xy, width, height)

            else:  # А핀ное преобразование bounding box
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = xy @ M.T  # Трансформация
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n,
                                                                                    8)  # Пресеклирование перспективы или аффинное преобразование

                # Создание новых bounding box
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # Обрезка
                new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
                new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            # Фильтрация кандидатов
            i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
            targets = targets[i]
            targets[:, 1:5] = new[i]

        return im, targets

    def copy_paste(im, labels, segments, p=0.5):
        # Реализация аугментации Copy-Paste https://arxiv.org/abs/2012.07177, метки в виде nx5 массива np.array(cls, xyxy)
        n = len(segments)
        if p and n:
            h, w, c = im.shape  # Высота, ширина, количество каналов
            im_new = np.zeros(im.shape, np.uint8)
            for j in random.sample(range(n), k=round(p * n)):
                l, s = labels[j], segments[j]
                box = w - l[3], l[2], w - l[1], l[4]
                ioa = bbox_ioa(box, labels[:, 1:5])  # Пересечение по площади
                if (ioa < 0.30).all():  # Позволяет до 30% заслоения существующих меток
                    labels = np.concatenate((labels, [[l[0], *box]]), 0)
                    segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                    cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

            result = cv2.flip(im, 1)  # Аугментация сегментов (отражение слева направо)
            i = cv2.flip(im_new, 1).astype(bool)
            im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # Отладка

        return im, labels, segments

    def cutout(im, labels, p=0.5):
        # Применение аугментации cutout изображения https://arxiv.org/abs/1708.04552
        if random.random() < p:
            h, w = im.shape[:2]
            scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # Доля размера изображения
            for s in scales:
                mask_h = random.randint(1, int(h * s))  # Создание случайных масок
                mask_w = random.randint(1, int(w * s))

                # Box
                xmin = max(0, random.randint(0, w) - mask_w // 2)
                ymin = max(0, random.randint(0, h) - mask_h // 2)
                xmax = min(w, xmin + mask_w)
                ymax = min(h, ymin + mask_h)

                # Применение случайной цветовой маски
                im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

                # Возврат непрозрачных меток
                if len(labels) and s > 0.03:
                    box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                    ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # Пересечение по площади
                    labels = labels[ioa < 0.60]  # Удалить метки, заслоенные более чем на 60%

        return labels

    def mixup(im, labels, im2, labels2):
        # Применение аугментации MixUp https://arxiv.org/pdf/1710.09412.pdf
        r = np.random.beta(32.0, 32.0)  # Коэффициент mixup, alpha=beta=32.0
        im = (im * r + im2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)
        return im, labels

    def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Вычисление кандидатов для bounding box: box1 до аугментации, box2 после аугментации, wh_thr (пиксели), соотношение сторон, отношение площадей
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # Соотношение сторон
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # Кандидаты

    def classify_albumentations(
            augment=True,
            size=224,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
            hflip=0.5,
            vflip=0.0,
            jitter=0.4,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            auto_aug=False):
        # Albumentations для классификации в YOLOv5 (необязательно, используется только если пакет установлен)
        prefix = colorstr('albumentations: ')
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            check_version(A.__version__, '1.0.3', hard=True)  # Требование к версии
            if augment:  # Изменение размера и обрезка
                T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
                if auto_aug:
                    # TODO: Реализовать AugMix, AutoAug & RandAug в albumentation
                    LOGGER.info(f'{prefix}автоматические аугментации в настоящее время не поддерживаются')
                else:
                    if hflip > 0:
                        T += [A.HorizontalFlip(p=hflip)]
                    if vflip > 0:
                        T += [A.VerticalFlip(p=vflip)]
                    if jitter > 0:
                        color_jitter = (float(
                            jitter),) * 3  # Повтор значения для яркости, контраста, насыщенности, 0 оттенок
                        T += [A.ColorJitter(*color_jitter, 0)]
            else:  # Использование фиксированного обрезка для набора оценки (репродуцируемость)
                T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
            T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Нормализация и преобразование в Tensor
            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
            return A.Compose(T)

        except ImportError:  # Пакет не установлен, пропустить
            LOGGER.warning(f'{prefix}⚠️ не найден, установить с помощью `pip install albumentations` (рекомендуется)')
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    def classify_transforms(size=224):
        # Преобразования для применения, если albumentations не установлен
        assert isinstance(size,
                          int), f'ERROR: classify_transforms size {size} должен быть целым числом, а не (list, tuple)'
        # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    class LetterBox:
        # Класс LetterBox для YOLOv5 для предобработки изображений, т.е. T.Compose([LetterBox(size), ToTensor()])
        def __init__(self, size=(640, 640), auto=False, stride=32):
            super().__init__()
            self.h, self.w = (size, size) if isinstance(size, int) else size
            self.auto = auto  # Передать целое максимальное значение размера, автоматически вычислить короткую сторону с использованием шага
            self.stride = stride  # Используется с auto

        def __call__(self, im):  # im = np.array HWC
            imh, imw = im.shape[:2]
            r = min(self.h / imh, self.w / imw)  # Соотношение нового/стараго
            h, w = round(imh * r), round(imw * r)  # Измененный размер изображения
            hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
            top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
            im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
            im_out[top:top + h, left:left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            return im_out

    class CenterCrop:
        # Класс CenterCrop для YOLOv5 для предобработки изображений, т.е. T.Compose([CenterCrop(size), ToTensor()])
        def __init__(self, size=640):
            super().__init__()
            self.h, self.w = (size, size) if isinstance(size, int) else size

        def __call__(self, im):  # im = np.array HWC
            imh, imw = im.shape[:2]
            m = min(imh, imw)  # Минимальная размерность
            top, left = (imh - m) // 2, (imw - m) // 2
            return cv2.resize(im[top:top + m, left:left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    class ToTensor:
        # Класс ToTensor для YOLOv5 для предобработки изображений, т.е. T.Compose([LetterBox(size), ToTensor()])
        def __init__(self, half=False):
            super().__init__()
            self.half = half

        def __call__(self, im):  # im = np.array HWC в порядке BGR
            im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC в CHW -> BGR в RGB -> непрерывный
            im = torch.from_numpy(im)  # в torch
            im = im.half() if self.half else im.float()  # uint8 в fp16/32
            im /= 255.0  # 0-255 в 0.0-1.0
            return im