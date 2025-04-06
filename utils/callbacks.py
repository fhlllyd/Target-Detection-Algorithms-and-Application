# YOLOv5 🚀 от Ultralytics, лицензия GPL-3.0
"""
Утилиты для автоматического подбора размера батча
"""

from copy import deepcopy

import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile


def check_train_batch_size(model, imgsz=640, amp=True):
    # Проверка размера батча для обучения YOLOv5
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)  # вычисление оптимального размера батча


def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    # Автоматическое оценивание лучшего размера батча для YOLOv5 для использования `fraction` доступной CUDA памяти
    # Использование:
    #     import torch
    #     from utils.autobatch import autobatch
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))

    # Проверка устройства
    prefix = colorstr('AutoBatch: ')
    LOGGER.info(f'{prefix}Вычисление оптимального размера батча для --imgsz {imgsz}')
    device = next(model.parameters()).device  # получение устройства модели
    if device.type == 'cpu':
        LOGGER.info(f'{prefix}CUDA не обнаружена, используется размер батча по умолчанию для CPU {batch_size}')
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f'{prefix} ⚠️ Требуется torch.backends.cudnn.benchmark=False, используется размер батча по умолчанию {batch_size}')
        return batch_size

    # Проверка CUDA памяти
    gb = 1 << 30  # байты в гигабайты (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # свойства устройства
    t = properties.total_memory / gb  # общая память в ГиБ
    r = torch.cuda.memory_reserved(device) / gb  # зарезервированная память в ГиБ
    a = torch.cuda.memory_allocated(device) / gb  # выделенная память в ГиБ
    f = t - (r + a)  # свободная память в ГиБ
    LOGGER.info(f'{prefix}{d} ({properties.name}) {t:.2f}Гб общего, {r:.2f}Гб зарезервировано, {a:.2f}Гб выделено, {f:.2f}Гб свободно')

    # Профилирование размеров батча
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
    except Exception as e:
        LOGGER.warning(f'{prefix}{e}')

    # Поиск решения
    y = [x[2] for x in results if x]  # память [2]
    p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # полиномиальная аппроксимация первой степени
    b = int((f * fraction - p[1]) / p[0])  # пересечение с осью Y (оптимальный размер батча)
    if None in results:  # некоторые размеры не прошли
        i = results.index(None)  # первый индекс неудачи
        if b >= batch_sizes[i]:  # пересечение с осью Y выше точки неудачи
            b = batch_sizes[max(i - 1, 0)]  # выбрать предыдущую безопасную точку
    if b < 1 or b > 1024:  # b вне безопасного диапазона
        b = batch_size
        LOGGER.warning(f'{prefix}ВНИМАНИЕ ⚠️ Обнаружена аномалия CUDA, рекомендуется перезапустить среду и повторить команду.')

    fraction = (np.polyval(p, b) + r + a) / t  # фактическая доля, предсказанная
    LOGGER.info(f'{prefix}Используется размер батча {b} для {d} {t * fraction:.2f}Гб/{t:.2f}Гб ({fraction * 100:.0f}%) ✅')
    return b