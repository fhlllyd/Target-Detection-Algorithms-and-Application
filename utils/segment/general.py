import cv2
import numpy as np
import torch
import torch.nn.functional as F


def crop_mask(masks, boxes):
    """
    "Обрезает" предсказанные маски, обнуляя все, что не находится в предсказанном ограничивающем прямоугольнике.
    Векторизовано Чонгом (спасибо Чонгу).

    Аргументы:
        - masks должен быть тензором размером [h, w, n] с масками
        - boxes должен быть тензором размером [n, 4] с координатами ограничивающих прямоугольников в относительной точковой форме
    """

    # Получаем размеры тензора маски (высота, ширина, количество масков)
    n, h, w = masks.shape
    # Разделяем координаты ограничивающих прямоугольников на четыре части (x1, y1, x2, y2)
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 имеет размер (1,1,n)
    # Создаем тензор с индексами столбцов
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # r имеет размер (1,w,1)
    # Создаем тензор с индексами строк
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # c имеет размер (h,1,1)

    # Возвращаем маски, умноженные на логический тензор, который определяет, какие пиксели находятся внутри ограничивающего прямоугольника
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    Обрезает маски после увеличения их размера.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], где n - количество масков после подавления немаксимумов
    bboxes: [n, 4], где n - количество масков после подавления немаксимумов
    shape: размер входного изображения, (h, w)

    Возвращает: h, w, n
    """

    # Получаем размеры тензора прототипов (количество каналов, высота, ширина)
    c, mh, mw = protos.shape  # CHW
    # Вычисляем маски на основе прототипов и входных масков, применяем сигмоидальную функцию и меняем размерность
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    # Увеличиваем размер маски до заданного размера
    masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    # Обрезаем маски по ограничивающим прямоугольникам
    masks = crop_mask(masks, bboxes)  # CHW
    # Возвращаем бинарные маски (значения больше 0.5 заменяются на True, остальные на False)
    return masks.gt_(0.5)


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Обрезает маски до увеличения их размера.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], где n - количество масков после подавления немаксимумов
    bboxes: [n, 4], где n - количество масков после подавления немаксимумов
    shape: размер входного изображения, (h, w)

    Возвращает: h, w, n
    """

    # Получаем размеры тензора прототипов (количество каналов, высота, ширина)
    c, mh, mw = protos.shape  # CHW
    # Получаем высоту и ширину входного изображения
    ih, iw = shape
    # Вычисляем маски на основе прототипов и входных масков, применяем сигмоидальную функцию и меняем размерность
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    # Клонируем ограничивающие прямоугольники
    downsampled_bboxes = bboxes.clone()
    # Масштабируем координаты ограничивающих прямоугольников в соответствии с размером прототипов
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    # Обрезаем маски по масштабированным ограничивающим прямоугольникам
    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    # Если нужно увеличить размер маски
    if upsample:
        # Увеличиваем размер маски до заданного размера
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    # Возвращаем бинарные маски (значения больше 0.5 заменяются на True, остальные на False)
    return masks.gt_(0.5)


def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """
    img1_shape: размер входного изображения для модели, [h, w]
    img0_shape: исходный размер изображения, [h, w, 3]
    masks: [h, w, num]
    """
    # Масштабируем координаты (xyxy) от размера im1_shape до размера im0_shape
    if ratio_pad is None:  # вычисляем на основе im0_shape
        # Вычисляем коэффициент масштабирования
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        # Вычисляем отступы по ширине и высоте
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        # Если отступы уже заданы
        pad = ratio_pad[1]
    # Получаем верхний и левый отступы
    top, left = int(pad[1]), int(pad[0])  # y, x
    # Получаем нижний и правый отступы
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    # Проверяем, что размерность маски правильная
    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" должна быть 2 или 3, но получено {len(masks.shape)}')
    # Обрезаем маски по отступам
    masks = masks[top:bottom, left:right]
    # masks = masks.permute(2, 0, 1).contiguous()
    # masks = F.interpolate(masks[None], im0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    # Изменяем размер маски до исходного размера изображения
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_NEAREST)

    # Если маска имеет размерность 2, добавляем третье измерение
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks


def mask_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] m1 означает количество предсказанных объектов
    mask2: [M, n] m2 означает количество истинных объектов
    Примечание: n означает image_w x image_h

    Возвращает: IoU масков, [N, M]
    """
    # Вычисляем пересечение масков
    intersection = torch.matmul(mask1, mask2.t()).clamp(0)
    # Вычисляем объединение масков
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    # Возвращаем IoU масков
    return intersection / (union + eps)


def masks_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] m1 означает количество предсказанных объектов
    mask2: [N, n] m2 означает количество истинных объектов
    Примечание: n означает image_w x image_h

    Возвращает: IoU масков, (N, )
    """
    # Вычисляем пересечение масков
    intersection = (mask1 * mask2).sum(1).clamp(0)  # (N, )
    # Вычисляем объединение масков
    union = (mask1.sum(1) + mask2.sum(1))[None] - intersection  # (area1 + area2) - intersection
    # Возвращаем IoU масков
    return intersection / (union + eps)


def masks2segments(masks, strategy='largest'):
    # Преобразует маски (n,160,160) в сегменты (n,xy)
    segments = []
    for x in masks.int().cpu().numpy().astype('uint8'):
        # Находим контуры маски
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # объединяем все сегменты
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # выбираем самый большой сегмент
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # если сегменты не найдены
        segments.append(c.astype('float32'))
    return segments