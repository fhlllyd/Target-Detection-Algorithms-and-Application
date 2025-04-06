# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from IPython.display import display
from PIL import Image
from torch.cuda import amp

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_notebook, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, smart_inference_mode

# Вычисление padding для получения 'same' формы выхода
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Если dilation > 1, вычисляем фактический размер ядра
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # Фактический размер ядра
    # Если padding не задан, вычисляем его автоматически
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # Автоматический padding
    return p

# Стандартная свертка
class Conv(nn.Module):
    # Стандартная свертка с аргументами (ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # Стандартная функция активации

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # Определение сверточного слоя
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # Определение слоя батч-нормализации
        self.bn = nn.BatchNorm2d(c2)
        # Определение функции активации
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # Прямой проход через свертку, батч-нормализацию и функцию активации
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # Прямой проход без батч-нормализации
        return self.act(self.conv(x))

# Глубинная свертка
class DWConv(Conv):
    # Глубинная свертка
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

# Глубинная транспонированная свертка
class DWConvTranspose2d(nn.ConvTranspose2d):
    # Глубинная транспонированная свертка
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))

# Слой трансформера
class TransformerLayer(nn.Module):
    # Слой трансформера https://arxiv.org/abs/2010.11929 (Слои LayerNorm удалены для лучшей производительности)
    def __init__(self, c, num_heads):
        super().__init__()
        # Определение линейных слоев для Q, K, V
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        # Определение слоя多头注意力
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        # Определение линейных слоев для полносвязных блоков
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        # Прямой проход через слой多头注意力 и полносвязные блоки
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x

# Блок трансформера
class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        # Определение сверточного слоя, если входные и выходные каналы различны
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        # Определение линейного слоя для позиционного вложения
        self.linear = nn.Linear(c2, c2)  # Учебное позиционное вложение
        # Определение последовательности слоев трансформера
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        # Прямой проход через сверточный слой (если есть), а затем через блок трансформера
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

# Стандартный бутылочное горлышко
class Bottleneck(nn.Module):
    # Стандартное бутылочное горлышко
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # Скрытые каналы
        # Определение сверточных слоев
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # Прямой проход через бутылочное горлышко
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# CSP Bottleneck
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # Скрытые каналы
        # Определение сверточных слоев
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        # Определение слоя батч-нормализации
        self.bn = nn.BatchNorm2d(2 * c_)  # Применяется к cat(cv2, cv3)
        # Определение функции активации
        self.act = nn.SiLU()
        # Определение последовательности бутылочных горлышек
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        # Прямой проход через CSP Bottleneck
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

# Кросс- свертка с понижением дискретизации
class CrossConv(nn.Module):
    # Кросс- свертка с понижением дискретизации
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # Скрытые каналы
        # Определение сверточных слоев
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # Прямой проход через кросс- свертку
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# CSP Bottleneck с тремя свертками
class C3(nn.Module):
    # CSP Bottleneck с тремя свертками
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, количество, shortcut, группы, расширение
        super().__init__()
        c_ = int(c2 * e)  # Скрытые каналы
        # Определение сверточных слоев
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # Опционально act=FReLU(c2)
        # Определение последовательности бутылочных горлышек
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        # Прямой проход через C3 модуль
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# C3 модуль с кросс- свертками
class C3x(C3):
    # C3 модуль с кросс- свертками
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # Определение последовательности кросс- сверток
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))

# C3 модуль с TransformerBlock()
class C3TR(C3):
    # C3 модуль с TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # Определение блока трансформера
        self.m = TransformerBlock(c_, c_, 4, n)

# C3 модуль с SPP()
class C3SPP(C3):
    # C3 модуль с SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # Определение слоя пространственного пирамидального пулинга
        self.m = SPP(c_, c_, k)

# C3 модуль с GhostBottleneck()
class C3Ghost(C3):
    # C3 модуль с GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # Скрытые каналы
        # Определение последовательности GhostBottleneck
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

# Spatial Pyramid Pooling (SPP) слой https://arxiv.org/abs/1406.4729
class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) слой https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # Скрытые каналы
        # Определение сверточных слоев
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        # Определение списка слоев максимального пулинга
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        # Прямой проход через SPP слой
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Подавление предупреждения torch 1.9.0 max_pool2d()
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

# Spatial Pyramid Pooling - Fast (SPPF) слой для YOLOv5 от Glenn Jocher
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) слой для YOLOv5 от Glenn Jocher
    def __init__(self, c1, c2, k=5):  # Эквивалентно SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # Скрытые каналы
        # Определение сверточных слоев
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        # Определение слоя максимального пулинга
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        # Прямой проход через SPPF слой
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Подавление предупреждения torch 1.9.0 max_pool2d()
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# Focus wh информацию в c-пространство
class Focus(nn.Module):
    # Focus wh информацию в c-пространство
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, ядро, шаг, padding, группы
        super().__init__()
        # Определение сверточного слоя
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # Прямой проход через Focus слой
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))

# Ghost Convolution https://github.com/huawei-noah/ghostnet
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, ядро, шаг, группы
        super().__init__()
        c_ = c2 // 2  # Скрытые каналы
        # Определение сверточных слоев
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        # Прямой проход через GhostConv слой
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

# Ghost Bottleneck https://github.com/huawei-noah/ghostnet
class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, ядро, шаг
        super().__init__()
        c_ = c2 // 2
        # Определение последовательности слоев GhostBottleneck
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        # Определение shortcut
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        # Прямой проход через GhostBottleneck
        return self.conv(x) + self.shortcut(x)

# Сжатие ширины-высоты в каналы, например, x(1,64,80,80) в x(1,256,40,40)
class Contract(nn.Module):
    # Сжатие ширины-высоты в каналы, например, x(1,64,80,80) в x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        # Прямой проход через Contract слой
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)

# Расширение каналов в ширину-высоту, например, x(1,64,80,80) в x(1,16,160,160)
class Expand(nn.Module):
    # Расширение каналов в ширину-высоту, например, x(1,64,80,80) в x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        # Прямой проход через Expand слой
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)

# Конкатенация списка тензоров по измерению
class Concat(nn.Module):
    # Конкатенация списка тензоров по измерению
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # Прямой проход через Concat слой
        return torch.cat(x, self.d)


# Класс YOLOv5 MultiBackend для Python-инференса на различных бэкендах
class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend класс для Python-инференса на различных бэкендах
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Использование:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # локальное импортирование, чтобы избежать циклического импорта

        super().__init__()
        # Преобразование весов в строку
        w = str(weights[0] if isinstance(weights, list) else weights)
        # Определение типа модели
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        # Установка режима FP16 для поддерживаемых типов моделей
        fp16 &= pt or jit or onnx or engine  # FP16
        # Проверка на форматы BHWC (в отличие от torch BCWH)
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC форматы (vs torch BCWH)
        # Установка значения шага по умолчанию
        stride = 32  # значение шага по умолчанию
        # Проверка доступности CUDA и типа устройства
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # использование CUDA
        # Загрузка весов, если они не находятся локально
        if not (pt or triton):
            w = attempt_download(w)  # загрузка, если не локально

        if pt:  # PyTorch
            # Загрузка модели PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            # Получение шага модели
            stride = max(int(model.stride.max()), 32)  # шаг модели
            # Получение имен классов
            names = model.module.names if hasattr(model, 'module') else model.names  # получение имен классов
            # Установка типа данных модели
            model.half() if fp16 else model.float()
            # Присвоение модели атрибуту
            self.model = model  # явное присвоение для to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            # Логирование загрузки модели TorchScript
            LOGGER.info(f'Загрузка {w} для TorchScript-инференса...')
            # Файл с метаданными модели
            extra_files = {'config.txt': ''}  # метаданные модели
            # Загрузка модели TorchScript
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            # Установка типа данных модели
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:  # загрузка словаря метаданных
                # Десериализация метаданных
                d = json.loads(extra_files['config.txt'],
                               object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                      for k, v in d.items()})
                # Получение шага и имен классов
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            # Логирование загрузки модели ONNX с использованием OpenCV DNN
            LOGGER.info(f'Загрузка {w} для ONNX OpenCV DNN-инференса...')
            # Проверка наличия необходимых библиотек
            check_requirements('opencv-python>=4.5.4')
            # Загрузка модели ONNX в OpenCV DNN
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            # Логирование загрузки модели ONNX с использованием ONNX Runtime
            LOGGER.info(f'Загрузка {w} для ONNX Runtime-инференса...')
            # Проверка наличия необходимых библиотек
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            # Определение провайдеров для ONNX Runtime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            # Создание сессии ONNX Runtime
            session = onnxruntime.InferenceSession(w, providers=providers)
            # Получение имен выходных тензоров
            output_names = [x.name for x in session.get_outputs()]
            # Получение метаданных модели
            meta = session.get_modelmeta().custom_metadata_map  # метаданные
            if 'stride' in meta:
                # Получение шага и имен классов из метаданных
                stride, names = int(meta['stride']), eval(meta['names'])
        elif xml:  # OpenVINO
            # Логирование загрузки модели OpenVINO
            LOGGER.info(f'Загрузка {w} для OpenVINO-инференса...')
            # Проверка наличия необходимых библиотек
            check_requirements('openvino')  # требуется openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch
            # Инициализация ядра OpenVINO
            ie = Core()
            if not Path(w).is_file():  # если не *.xml
                # Поиск файла *.xml в директории
                w = next(Path(w).glob('*.xml'))  # получение файла *.xml из директории *_openvino_model
            # Чтение модели OpenVINO
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                # Установка разметки входного тензора
                network.get_parameters()[0].set_layout(Layout("NCHW"))
            # Получение размерности батча
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                # Получение размера батча
                batch_size = batch_dim.get_length()
            # Компиляция модели OpenVINO
            executable_network = ie.compile_model(network, device_name="CPU")  # device_name="MYRIAD" для Intel NCS2
            # Загрузка метаданных
            stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # загрузка метаданных
        elif engine:  # TensorRT
            # Логирование загрузки модели TensorRT
            LOGGER.info(f'Загрузка {w} для TensorRT-инференса...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            # Проверка версии TensorRT
            check_version(trt.__version__, '7.0.0', hard=True)  # требуется tensorrt>=7.0.0
            if device.type == 'cpu':
                # Установка устройства на GPU, если используется CPU
                device = torch.device('cuda:0')
            # Определение именованного кортежа для привязок
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            # Инициализация логгера TensorRT
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                # Десериализация модели TensorRT
                model = runtime.deserialize_cuda_engine(f.read())
            # Создание контекста выполнения
            context = model.create_execution_context()
            # Инициализация упорядоченного словаря для привязок
            bindings = OrderedDict()
            # Инициализация списка имен выходных тензоров
            output_names = []
            fp16 = False  # значение по умолчанию, обновляется ниже
            dynamic = False
            for i in range(model.num_bindings):
                # Получение имени привязки
                name = model.get_binding_name(i)
                # Получение типа данных привязки
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # динамический размер
                        dynamic = True
                        # Установка динамического размера входного тензора
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # выход
                    # Добавление имени выходного тензора в список
                    output_names.append(name)
                # Получение размера привязки
                shape = tuple(context.get_binding_shape(i))
                # Создание тензора PyTorch из пустого массива NumPy
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                # Добавление привязки в упорядоченный словарь
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            # Создание упорядоченного словаря с адресами привязок
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            # Получение размера батча
            batch_size = bindings['images'].shape[0]  # если динамический, это максимальный размер батча
        elif coreml:  # CoreML
            # Логирование загрузки модели CoreML
            LOGGER.info(f'Загрузка {w} для CoreML-инференса...')
            import coremltools as ct
            # Загрузка модели CoreML
            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            # Логирование загрузки модели TensorFlow SavedModel
            LOGGER.info(f'Загрузка {w} для TensorFlow SavedModel-инференса...')
            import tensorflow as tf
            keras = False  # предполагается TF1 saved_model
            # Загрузка модели TensorFlow SavedModel
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            # Логирование загрузки модели TensorFlow GraphDef
            LOGGER.info(f'Загрузка {w} для TensorFlow GraphDef-инференса...')
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                # Обертка замороженного графа TensorFlow
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # обернуто
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                # Получение имен выходных тензоров из графа TensorFlow
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, 'rb') as f:
                # Чтение графа TensorFlow из файла
                gd.ParseFromString(f.read())
            # Создание замороженной функции
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                # Логирование загрузки модели TensorFlow Lite Edge TPU
                LOGGER.info(f'Загрузка {w} для TensorFlow Lite Edge TPU-инференса...')
                delegate = {
                    'Linux': 'libedgetpu.so.1',
                    'Darwin': 'libedgetpu.1.dylib',
                    'Windows': 'edgetpu.dll'}[platform.system()]
                # Создание интерпретатора TensorFlow Lite Edge TPU
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                # Логирование загрузки модели TensorFlow Lite
                LOGGER.info(f'Загрузка {w} для TensorFlow Lite-инференса...')
                # Создание интерпретатора TensorFlow Lite
                interpreter = Interpreter(model_path=w)  # загрузка модели TFLite
            # Выделение памяти для тензоров
            interpreter.allocate_tensors()  # выделение памяти
            # Получение информации о входных тензорах
            input_details = interpreter.get_input_details()  # входы
            # Получение информации о выходных тензорах
            output_details = interpreter.get_output_details()  # выходы
            # Загрузка метаданных
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta['stride']), meta['names']
        elif tfjs:  # TF.js
            # Выбрасывание исключения, если используется TF.js
            raise NotImplementedError('ОШИБКА: YOLOv5 TF.js-инференс не поддерживается')
        elif paddle:  # PaddlePaddle
            # Логирование загрузки модели PaddlePaddle
            LOGGER.info(f'Загрузка {w} для PaddlePaddle-инференса...')
            # Проверка наличия необходимых библиотек
            check_requirements('paddlepaddle-gpu' if cuda else 'paddlepaddle')
            import paddle.inference as pdi
            if not Path(w).is_file():  # если не *.pdmodel
                # Поиск файла *.pdmodel в директории
                w = next(Path(w).rglob('*.pdmodel'))  # получение файла *.pdmodel из директории *_paddle_model
            # Получение пути к весам модели
            weights = Path(w).with_suffix('.pdiparams')
            # Создание конфигурации PaddlePaddle
            config = pdi.Config(str(w), str(weights))
            if cuda:
                # Включение использования GPU
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            # Создание предиктора PaddlePaddle
            predictor = pdi.create_predictor(config)
            # Получение хэндла входного тензора
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            # Получение имен выходных тензоров
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            # Логирование использования сервера NVIDIA Triton Inference Server
            LOGGER.info(f'Использование {w} как сервера NVIDIA Triton Inference Server...')
            # Проверка наличия необходимых библиотек
            check_requirements('tritonclient[all]')
            from utils.triton import TritonRemoteModel
            # Создание удаленной модели Triton
            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            # Выбрасывание исключения, если формат модели не поддерживается
            raise NotImplementedError(f'ОШИБКА: {w} не является поддерживаемым форматом')

            # class names
            if 'names' not in locals():
                # Загрузка имен классов из файла data или использование дефолтных
                names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
                # Если модель ImageNet - загрузить читаемые имена
                if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
                    names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # человеческие имена

            # Присвоение всех переменных в self
            self.__dict__.update(locals())  # assign all variables to self

        def forward(self, im, augment=False, visualize=False):
            # YOLOv5 MultiBackend inference
            b, ch, h, w = im.shape  # batch, channel, height, width
            # Преобразование в FP16 если нужно
            if self.fp16 and im.dtype != torch.float16:
                im = im.half()  # to FP16
            # Преобразование формата BHWC если требуется
            if self.nhwc:
                im = im.permute(0, 2, 3, 1)  # torch BCHW -> numpy BHWC shape(1,320,192,3)

            # PyTorch inference
            if self.pt:  # PyTorch
                y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
            # TorchScript inference
            elif self.jit:  # TorchScript
                y = self.model(im)
            # ONNX OpenCV DNN inference
            elif self.dnn:  # ONNX OpenCV DNN
                im = im.cpu().numpy()  # torch -> numpy
                self.net.setInput(im)
                y = self.net.forward()
            # ONNX Runtime inference
            elif self.onnx:  # ONNX Runtime
                im = im.cpu().numpy()  # torch -> numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            # OpenVINO inference
            elif self.xml:  # OpenVINO
                im = im.cpu().numpy()  # FP32
                y = list(self.executable_network([im]).values())
            # TensorRT inference
            elif self.engine:  # TensorRT
                # Если динамический размер - обновляем
                if self.dynamic and im.shape != self.bindings['images'].shape:
                    i = self.model.get_binding_index('images')
                    self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                    self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                    # Обновляем размеры выходных тензоров
                    for name in self.output_names:
                        i = self.model.get_binding_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
                # Проверка размеров
                s = self.bindings['images'].shape
                assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
                # Устанавливаем адрес входного тензора
                self.binding_addrs['images'] = int(im.data_ptr())
                # Выполняем инференс
                self.context.execute_v2(list(self.binding_addrs.values()))
                # Собираем выходные данные
                y = [self.bindings[x].data for x in sorted(self.output_names)]
            # CoreML inference
            elif self.coreml:  # CoreML
                im = im.cpu().numpy()
                im = Image.fromarray((im[0] * 255).astype('uint8'))
                # im = im.resize((192, 320), Image.ANTIALIAS)
                y = self.model.predict({'image': im})  # координаты xywh нормализованные
                # Обработка выходных данных
                if 'confidence' in y:
                    box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy пиксели
                    conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                    y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
                else:
                    y = list(reversed(y.values()))  # для сегментационных моделей (pred, proto)
            # PaddlePaddle inference
            elif self.paddle:  # PaddlePaddle
                im = im.cpu().numpy().astype(np.float32)
                self.input_handle.copy_from_cpu(im)
                self.predictor.run()
                y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
            # NVIDIA Triton inference
            elif self.triton:  # NVIDIA Triton Inference Server
                y = self.model(im)
            # TensorFlow inference (SavedModel, GraphDef, Lite, Edge TPU)
            else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
                im = im.cpu().numpy()
                if self.saved_model:  # SavedModel
                    y = self.model(im, training=False) if self.keras else self.model(im)
                elif self.pb:  # GraphDef
                    y = self.frozen_func(x=self.tf.constant(im))
                else:  # Lite or Edge TPU
                    input = self.input_details[0]
                    int8 = input['dtype'] == np.uint8  # проверка квантованной модели
                    if int8:
                        scale, zero_point = input['quantization']
                        im = (im / scale + zero_point).astype(np.uint8)  # дескейлинг
                    self.interpreter.set_tensor(input['index'], im)
                    self.interpreter.invoke()
                    y = []
                    # Обработка выходных тензоров
                    for output in self.output_details:
                        x = self.interpreter.get_tensor(output['index'])
                        if int8:
                            scale, zero_point = output['quantization']
                            x = (x.astype(np.float32) - zero_point) * scale  # ресейлинг
                        y.append(x)
                # Преобразование координат в пиксели
                y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
                y[0][..., :4] *= [w, h, w, h]  # xywh нормализованные -> пиксели

            # Обработка выходных данных
            if isinstance(y, (list, tuple)):
                return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
            else:
                return self.from_numpy(y)

        def from_numpy(self, x):
            # Преобразование из numpy в torch
            return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

        def warmup(self, imgsz=(1, 3, 640, 640)):
            # Топливо для модели (런 1 inference)
            warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
            if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
                im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # вход
                for _ in range(2 if self.jit else 1):  #
                    self.forward(im)  # топливо

        @staticmethod
        def _model_type(p='path/to/model.pt'):
            # Определение типа модели по пути
            from export import export_formats
            from utils.downloads import is_url
            sf = list(export_formats().Suffix)  # расширения
            if not is_url(p, check=False):
                check_suffix(p, sf)  # проверка расширения
            url = urlparse(p)  # если URL - возможно Triton
            types = [s in Path(p).name for s in sf]
            types[8] &= not types[9]  # tflite &= not edgetpu
            triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
            return types + [triton]

        @staticmethod
        def _load_metadata(f=Path('path/to/meta.yaml')):
            # Загрузка метаданных из meta.yaml
            if f.exists():
                d = yaml_load(f)
                return d['stride'], d['names']  # возвращает stride и names
            return None, None

class AutoShape(nn.Module):
    # Обертка YOLOv5 для обрабатывания разнообразных входных данных (cv2/np/PIL/torch). Включает предобработку, инференс и NMS
    conf = 0.25  # Порог уверенности NMS
    iou = 0.45  # Порог IoU NMS
    agnostic = False  # Класс-агностичный NMS
    multi_label = False  # Множественные метки на бокс
    classes = None  # (опциональный список) фильтрация по классам, например [0, 15, 16] для COCO (люди, кошки, собаки)
    max_det = 1000  # Максимальное количество детекций на изображение
    amp = False  # Автоматическая смешанная точность (AMP) для инференса

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Добавление AutoShape... ')
        # Копирование атрибутов из модели
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())
        self.dmb = isinstance(model, DetectMultiBackend)  # Экземпляр DetectMultiBackend
        self.pt = not self.dmb or model.pt  # PyTorch модель
        self.model = model.eval()
        if self.pt:
            # Получение последнего слоя Detect
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]
            m.inplace = False  # Без inplace-операций для безопасности в многопоточной среде
            m.export = True  # Не выводить значения потерь

    def _apply(self, fn):
        # Применение to(), cpu(), cuda(), half() к тензорам модели, не являющимся параметрами или буферами
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        # Инференс с различных источников. Для size(height=640, width=1280) примеры входов:
        #   файл:        ims = 'data/images/zidane.jpg'
        #   URI:          = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:       = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR → RGB
        #   PIL:          = Image.open('image.jpg')
        #   numpy:        = np.zeros((640,1280,3))  # HWC
        #   torch:        = torch.zeros(16,3,320,640)  # BCHW (масштаб 0-1)
        #   список:       = [Image.open('image1.jpg'), ...]

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)
            autocast = self.amp and (p.device.type != 'cpu')
            if isinstance(ims, torch.Tensor):  # Torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)

            # Предобработка
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])
            shape0, shape1, files = [], [], []
            for i, im in enumerate(ims):
                f = f'image{i}'
                if isinstance(im, (str, Path)):  # Имя файла или URI
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL изображение
                    im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
                files.append(Path(f).with_suffix('.jpg').name)
                if im.shape[0] < 5:  # CHW → HWC
                    im = im.transpose((1, 2, 0))
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # Трехканальное изображение
                s = im.shape[:2]
                shape0.append(s)
                g = max(size) / max(s)  # Коэффициент масштабирования
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # Форма для инференса
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # Padding
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # BHWC → BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 → fp16/32

        with amp.autocast(autocast):
            # Инференс
            with dt[1]:
                y = self.model(x, augment=augment)

            # Постобработка
            with dt[2]:
                y = non_max_suppression(y if self.dmb else y[0],
                                        self.conf,
                                        self.iou,
                                        self.classes,
                                        self.agnostic,
                                        self.multi_label,
                                        max_det=self.max_det)
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    # Класс для хранения результатов инференса YOLOv5
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # Нормализация
        self.ims = ims  # Список изображений в формате numpy
        self.pred = pred  # Список тензоров (xyxy, conf, cls)
        self.names = names  # Имена классов
        self.files = files  # Имена файлов изображений
        self.times = times  # Время профилирования
        self.xyxy = pred  # Координаты в формате xyxy (пиксели)
        self.xywh = [xyxy2xywh(x) for x in pred]  # Координаты в формате xywh (пиксели)
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # Нормализованные xyxy
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # Нормализованные xywh
        self.n = len(self.pred)  # Количество изображений (размер батча)
        self.t = tuple(x.t / self.n * 1E3 for x in times)  # Время в миллисекундах
        self.s = tuple(shape)  # Форма BCHW для инференса

        def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
            s, crops = '', []
            for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
                s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
                if pred.shape[0]:
                    # Подсчет количества детекций по классам
                    for c in pred[:, -1].unique():
                        n = (pred[:, -1] == c).sum()
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                    s = s.rstrip(', ')
                    if show or save or render or crop:
                        # Создание объекта для рисования
                        annotator = Annotator(im, example=str(self.names))
                        # Рендеринг каждого бокса
                        for *box, conf, cls in reversed(pred):
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            if crop:
                                # Сохранение вырезов
                                file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                                crops.append({
                                    'box': box,
                                    'conf': conf,
                                    'cls': cls,
                                    'label': label,
                                    'im': save_one_box(box, im, file=file, save=save)})
                            else:
                                # Отображение меток
                                annotator.box_label(box, label if labels else '', color=colors(cls))
                        im = annotator.im
                else:
                    s += '(no detections)'

                # Преобразование изображения
                im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im
                if show:
                    # Отображение изображения
                    display(im) if is_notebook() else im.show(self.files[i])
                if save:
                    # Сохранение изображения
                    f = self.files[i]
                    im.save(save_dir / f)
                    if i == self.n - 1:
                        LOGGER.info(
                            f"Сохранено {self.n} изображение{'й' * (self.n > 1)} в {colorstr('bold', save_dir)}")
                if render:
                    # Сохранение в виде массива
                    self.ims[i] = np.asarray(im)
            if pprint:
                # Форматированный вывод
                s = s.lstrip('\n')
                return f'{s}\nСкорость: %.1fms предобработка, %.1fms инференс, %.1fms NMS на изображение при размере {self.s}' % self.t
            if crop:
                if save:
                    LOGGER.info(f'Результаты вырезок сохранены в {save_dir}\n')
                return crops

        @TryExcept('Showing images is not supported in this environment')
        def show(self, labels=True):
            # Метод для отображения результатов
            self._run(show=True, labels=labels)  # show results

        def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
            # Метод для сохранения результатов
            save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
            self._run(save=True, labels=labels, save_dir=save_dir)  # save results

        def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
            # Метод для вырезания объектов
            save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
            return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

        def render(self, labels=True):
            # Метод для отрисовки результатов
            self._run(render=True, labels=labels)  # render results
            return self.ims

        def pandas(self):
            # Преобразование в pandas DataFrame
            new = copy(self)  # return copy
            ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
            cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
            for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
                # Преобразование координат
                a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]
                setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
            return new

        def tolist(self):
            # Преобразование в список объектов Detections
            r = range(self.n)  # iterable
            x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
            return x

        def print(self):
            # Печать результатов
            LOGGER.info(self.__str__())

        def __len__(self):  # override len(results)
            return self.n

        def __str__(self):  # override print(results)
            return self._run(pprint=True)  # print results

        def __repr__(self):
            return f'YOLOv5 {self.__class__} instance\n' + self.__str__()

    class Proto(nn.Module):
        # Модуль для генерации прототипов для сегментации
        def __init__(self, c1, c_=256, c2=32):  # ch_in, количество прототипов, количество масок
            super().__init__()
            self.cv1 = Conv(c1, c_, k=3)
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.cv2 = Conv(c_, c_, k=3)
            self.cv3 = Conv(c_, c2)

        def forward(self, x):
            # Прямой проход через модуль
            return self.cv3(self.cv2(self.upsample(self.cv1(x))))

    class Classify(nn.Module):
        # Голова для классификации YOLOv5
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
            super().__init__()
            c_ = 1280  # Размер для efficientnet_b0
            self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
            self.pool = nn.AdaptiveAvgPool2d(1)  # Приведение к размеру (b,c_,1,1)
            self.drop = nn.Dropout(p=0.0, inplace=True)
            self.linear = nn.Linear(c_, c2)  # Полносвязный слой

        def forward(self, x):
            # Прямой проход через голову
            if isinstance(x, list):
                x = torch.cat(x, 1)
            return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))