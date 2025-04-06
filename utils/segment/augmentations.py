# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Функции аугментации изображений
"""
import math
import random
import cv2
import numpy as np
from ..augmentations import box_candidates
from ..general import resample_segments, segment2box


def mixup(im, labels, segments, im2, labels2, segments2):
    # Применяет аугментацию MixUp https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # коэффициент смешивания, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    segments = np.concatenate((segments, segments2), 0)
    return im, labels, segments


def random_perspective(im,
                       targets=(),
                       segments=(),
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Центрирование
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x - сдвиг (пиксели)
    C[1, 2] = -im.shape[0] / 2  # y - сдвиг (пиксели)

    # Перспектива
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x - перспектива (вокруг y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y - перспектива (вокруг x)

    # Поворот и Масштабирование
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # добавить вращения на 90 градусов к небольшим вращениям
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Сдвиг (Shear)
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x - сдвиг (градусы)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y - сдвиг (градусы)

    # Перемещение (Translation)
    T = np.eye(3)
    T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)  # x - перемещение (пиксели)
    T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)  # y - перемещение (пиксели)

    # Комбинированная матрица вращения
    M = T @ S @ R @ P @ C  # порядок операций (справа налево) важен
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # если изображение изменилось
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # аффинное преобразование
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Визуализация
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # исходное
    # ax[1].imshow(im2[:, :, ::-1])  # преобразованное

    # Преобразование координат меток
    n = len(targets)
    new_segments = []
    if n:
        new = np.zeros((n, 4))
        segments = resample_segments(segments)  # апсемплирование
        for i, segment in enumerate(segments):
            xy = np.ones((len(segment), 3))
            xy[:, :2] = segment
            xy = xy @ M.T  # преобразование
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2])  # масштабирование перспективы или аффинное преобразование
            # обрезка
            new[i] = segment2box(xy, width, height)
            new_segments.append(xy)
        # фильтрация кандидатов
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01)
        targets = targets[i]
        targets[:, 1:5] = new[i]
        new_segments = np.array(new_segments)[i]
    return im, targets, new_segments
