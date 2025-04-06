# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
import math

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download


class Sum(nn.Module):
    # Взвешенная сумма 2 или более слоев https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: количество входов
        super().__init__()
        self.weight = weight  # применять веса
        self.iter = range(n - 1)  # итератор
        if weight:
            # Инициализация весов
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)

    def forward(self, x):
        y = x[0]  # без веса
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y += x[i + 1] * w[i]
        else:
            for i in self.iter:
                y += x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Смешанная глубинная свертка https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, ядра, шаг, стратегия каналов
        super().__init__()
        n = len(k)  # количество сверток
        if equal_ch:
            # Равное распределение каналов
            i = torch.linspace(0, n - 1E-6, c2).floor()
            c_ = [(i == g).sum() for g in range(n)]
        else:
            # Равное количество параметров
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()

        # Создание модулей
        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False)
            for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ансамбль моделей
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        # Инференс для каждого модели в ансамбле
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # максимальный ансамбль
        # y = torch.stack(y).mean(0)  # средний ансамбль
        y = torch.cat(y, 1)  # NMS ансамбль
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Загрузка ансамбля моделей weights=[a,b,c] или одиночной модели
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu')  # загрузка чекпоинта
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 модель

        # Обновления совместимости
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # преобразование в словарь

        # Добавление модели в ансамбль
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())

    # Обновления для модулей
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # совместимость с torch 1.7.0
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # совместимость с torch 1.11.0

    # Возвращение модели
    if len(model) == 1:
        return model[-1]

    # Возвращение детектирующего ансамбля
    print(f'Ансамбль создан с {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # максимальный шаг
    assert all(model[0].nc == m.nc for m in model), f'Модели имеют разные количества классов: {[m.nc for m in model]}'
    return model