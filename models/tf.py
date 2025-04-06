# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
TensorFlow, Keras and TFLite versions of YOLOv5
Authored by https://github.com/zldrobit in PR https://github.com/ultralytics/yolov5/pull/1127

Usage:
    $ python models/tf.py --weights yolov5s.pt

Export:
    $ python export.py --weights yolov5s.pt --include saved_model pb tflite tfjs
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Директория YOLOv5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Добавление ROOT в PATH

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow import keras

from models.common import (C3, SPP, SPPF, Bottleneck, BottleneckCSP, C3x, Concat, Conv, CrossConv, DWConv,
                           DWConvTranspose2d, Focus, autopad)
from models.experimental import MixConv2d, attempt_load
from models.yolo import Detect, Segment
from utils.activations import SiLU
from utils.general import LOGGER, make_divisible, print_args


class TFBN(keras.layers.Layer):
    # Обертка для TensorFlow BatchNormalization
    def __init__(self, w=None):
        super().__init__()
        # Инициализация слоя с весами из PyTorch
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w.bias.numpy()),
            gamma_initializer=keras.initializers.Constant(w.weight.numpy()),
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.numpy()),
            moving_variance_initializer=keras.initializers.Constant(w.running_var.numpy()),
            epsilon=w.eps)

    def call(self, inputs):
        return self.bn(inputs)


class TFPad(keras.layers.Layer):
    # Паддинг в пространственных измерениях 1 и 2
    def __init__(self, pad):
        super().__init__()
        # Преобразование паддинга в формат TensorFlow
        if isinstance(pad, int):
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        else:  # tuple/list
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])

    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode='constant', constant_values=0)


class TFConv(keras.layers.Layer):
    # Стандартная свертка
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        # ch_in, ch_out, веса, ядро, шаг, паддинг, группы
        super().__init__()
        assert g == 1, "TF v2.2 Conv2D не поддерживает 'groups'"
        # Конфигурация свертки с учетом паддинга
        conv = keras.layers.Conv2D(
            filters=c2,
            kernel_size=k,
            strides=s,
            padding='SAME' if s == 1 else 'VALID',
            use_bias=not hasattr(w, 'bn'),
            kernel_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer='zeros' if hasattr(w, 'bn') else keras.initializers.Constant(w.conv.bias.numpy()))
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        self.bn = TFBN(w.bn) if hasattr(w, 'bn') else tf.identity
        self.act = activations(w.act) if act else tf.identity

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs)))


class TFDWConv(keras.layers.Layer):
    # Глубинная свертка
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True, w=None):
        # ch_in, ch_out, веса, ядро, шаг, паддинг, группы
        super().__init__()
        assert c2 % c1 == 0, f'Выходной размер {c2} должен быть кратным {c1}'
        # Конфигурация глубинной свертки
        conv = keras.layers.DepthwiseConv2D(
            kernel_size=k,
            depth_multiplier=c2 // c1,
            strides=s,
            padding='SAME' if s == 1 else 'VALID',
            use_bias=not hasattr(w, 'bn'),
            depthwise_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer='zeros' if hasattr(w, 'bn') else keras.initializers.Constant(w.conv.bias.numpy()))
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        self.bn = TFBN(w.bn) if hasattr(w, 'bn') else tf.identity
        self.act = activations(w.act) if act else tf.identity

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs)))


class TFDWConvTranspose2d(keras.layers.Layer):
    # Глубинная транспонированная свертка
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0, w=None):
        # ch_in, ch_out, веса, ядро, шаг, паддинг, группы
        super().__init__()
        assert c1 == c2, 'Входной и выходной размеры должны совпадать'
        assert k == 4 and p1 == 1, 'Только k=4 и p1=1 поддерживаются'
        # Разбивка весов для каждого канала
        weight, bias = w.weight.permute(2, 3, 1, 0).numpy(), w.bias.numpy()
        self.c1 = c1
        self.conv = [
            keras.layers.Conv2DTranspose(filters=1,
                                         kernel_size=k,
                                         strides=s,
                                         padding='VALID',
                                         output_padding=p2,
                                         use_bias=True,
                                         kernel_initializer=keras.initializers.Constant(weight[..., i:i + 1]),
                                         bias_initializer=keras.initializers.Constant(bias[i])) for i in range(c1)]

    def call(self, inputs):
        return tf.concat([m(x) for m, x in zip(self.conv, tf.split(inputs, self.c1, 3))], 3)[:, 1:-1, 1:-1]


class TFFocus(keras.layers.Layer):
    # Сжатие пространственных размеров в каналы
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        # ch_in, ch_out, ядро, шаг, паддинг, группы
        super().__init__()
        # Сверточный слой для обработки сжатых данных
        self.conv = TFConv(c1 * 4, c2, k, s, p, g, act, w.conv)

    def call(self, inputs):  # x(b,w,h,c) -> y(b,w/2,h/2,4c)
        # Сжатие изображения путем дексимации
        inputs = [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :], inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]]
        return self.conv(tf.concat(inputs, 3))


class TFBottleneck(keras.layers.Layer):
    # Стандартный бутылочный слой
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):  # ch_in, ch_out, shortcut, группы, расширение
        super().__init__()
        c_ = int(c2 * e)  # Скрытые каналы
        # Сверточные слои бутылочного слоя
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFCrossConv(keras.layers.Layer):
    # Кросс-свёртка
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, w=None):
        super().__init__()
        c_ = int(c2 * e)  # Скрытые каналы
        # Сверточные слои для горизонтального и вертикального разбиения
        self.cv1 = TFConv(c1, c_, (1, k), (1, s), w=w.cv1)
        self.cv2 = TFConv(c_, c2, (k, 1), (s, 1), g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))

class TFConv2d(keras.layers.Layer):
    # Замена для PyTorch nn.Conv2D
    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None):
        super().__init__()
        assert g == 1, "TF v2.2 Conv2D не поддерживает 'groups'"
        # Слой свертки с весами из PyTorch
        self.conv = keras.layers.Conv2D(filters=c2,
                                        kernel_size=k,
                                        strides=s,
                                        padding='VALID',
                                        use_bias=bias,
                                        kernel_initializer=keras.initializers.Constant(
                                            w.weight.permute(2, 3, 1, 0).numpy()),
                                        bias_initializer=keras.initializers.Constant(w.bias.numpy()) if bias else None)

    def call(self, inputs):
        return self.conv(inputs)


class TFBottleneckCSP(keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, количество, shortcut, группы, расширение
        super().__init__()
        c_ = int(c2 * e)  # Скрытые каналы
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv2d(c1, c_, 1, 1, bias=False, w=w.cv2)
        self.cv3 = TFConv2d(c_, c_, 1, 1, bias=False, w=w.cv3)
        self.cv4 = TFConv(2 * c_, c2, 1, 1, w=w.cv4)
        self.bn = TFBN(w.bn)
        self.act = lambda x: keras.activations.swish(x)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        y1 = self.cv3(self.m(self.cv1(inputs)))
        y2 = self.cv2(inputs)
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))


class TFC3(keras.layers.Layer):
    # CSP Bottleneck с тремя свертками
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, количество, shortcut, группы, расширение
        super().__init__()
        c_ = int(c2 * e)  # Скрытые каналы
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFC3x(keras.layers.Layer):
    # C3 модуль с кросс-свёртками
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, количество, shortcut, группы, расширение
        super().__init__()
        c_ = int(c2 * e)  # Скрытые каналы
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([
            TFCrossConv(c_, c_, k=3, s=1, g=g, e=1.0, shortcut=shortcut, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFSPP(keras.layers.Layer):
    # Spatial pyramid pooling слой из YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), w=None):
        super().__init__()
        c_ = c1 // 2  # Скрытые каналы
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding='SAME') for x in k]

    def call(self, inputs):
        x = self.cv1(inputs)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))


class TFSPPF(keras.layers.Layer):
    # Spatial pyramid pooling-Fast слой
    def __init__(self, c1, c2, k=5, w=None):
        super().__init__()
        c_ = c1 // 2  # Скрытые каналы
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2)
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding='SAME')

    def call(self, inputs):
        x = self.cv1(inputs)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3))


class TFDetect(keras.layers.Layer):
    # TF YOLOv5 детект слой
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):  # detection layer
        super().__init__()
        self.stride = tf.convert_to_tensor(w.stride.numpy(), dtype=tf.float32)
        self.nc = nc  # Количество классов
        self.no = nc + 5  # Количество выходов на якорь
        self.nl = len(anchors)  # Количество детект слоев
        self.na = len(anchors[0]) // 2  # Количество якорных боксов
        self.grid = [tf.zeros(1)] * self.nl  # Инициализация сетки
        self.anchors = tf.convert_to_tensor(w.anchors.numpy(), dtype=tf.float32)
        self.anchor_grid = tf.reshape(self.anchors * tf.reshape(self.stride, [self.nl, 1, 1]), [self.nl, 1, -1, 1, 2])
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]
        self.training = False  # Установить False после построения модели
        self.imgsz = imgsz
        for i in range(self.nl):
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            self.grid[i] = self._make_grid(nx, ny)

    def call(self, inputs):
        z = []  # Выход для инференса
        x = []
        for i in range(self.nl):
            x.append(self.m[i](inputs[i]))
            # Преобразование x(bs,20,20,255) → x(bs,3,20,20,85)
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            x[i] = tf.reshape(x[i], [-1, ny * nx, self.na, self.no])

            if not self.training:  # Инференс
                y = x[i]
                grid = tf.transpose(self.grid[i], [0, 2, 1, 3]) - 0.5
                anchor_grid = tf.transpose(self.anchor_grid[i], [0, 2, 1, 3]) * 4
                xy = (tf.sigmoid(y[..., 0:2]) * 2 + grid) * self.stride[i]  # Координаты xy
                wh = tf.sigmoid(y[..., 2:4]) ** 2 * anchor_grid  # Размеры wh
                # Нормализация координат
                xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                y = tf.concat([xy, wh, tf.sigmoid(y[..., 4:5 + self.nc]), y[..., 5 + self.nc:]], -1)
                z.append(tf.reshape(y, [-1, self.na * ny * nx, self.no]))

        return tf.transpose(x, [0, 2, 1, 3]) if self.training else (tf.concat(z, 1),)

        @staticmethod
        def _make_grid(nx=20, ny=20):
            # Создание сетки для координат
            xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
            return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)

    class TFSegment(TFDetect):
        # Голова для сегментации YOLOv5
        def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), imgsz=(640, 640), w=None):
            super().__init__(nc, anchors, ch, imgsz, w)
            self.nm = nm  # Количество масок
            self.npr = npr  # Количество прототипов
            self.no = 5 + nc + self.nm  # Выходы на якорь
            self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]  # Выходной слой
            self.proto = TFProto(ch[0], self.npr, self.nm, w=w.proto)  # Прототипы
            self.detect = TFDetect.call

        def call(self, x):
            p = self.proto(x[0])
            # p = TFUpsample(None, scale_factor=4, mode='nearest')(self.proto(x[0]))  # Полноразмерные прототипы
            p = tf.transpose(p, [0, 3, 1, 2])  # Меняем размерность
            x = self.detect(self, x)
            return (x, p) if self.training else (x[0], p)

    class TFProto(keras.layers.Layer):

        def __init__(self, c1, c_=256, c2=32, w=None):
            super().__init__()
            self.cv1 = TFConv(c1, c_, k=3, w=w.cv1)
            self.upsample = TFUpsample(None, scale_factor=2, mode='nearest')
            self.cv2 = TFConv(c_, c_, k=3, w=w.cv2)
            self.cv3 = TFConv(c_, c2, w=w.cv3)

        def call(self, inputs):
            return self.cv3(self.cv2(self.upsample(self.cv1(inputs))))

    class TFUpsample(keras.layers.Layer):
        # TF версия torch.nn.Upsample()
        def __init__(self, size, scale_factor, mode, w=None):  # Все аргументы необходимы
            super().__init__()
            assert scale_factor % 2 == 0, "Масштаб должен быть кратным 2"
            self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * scale_factor, x.shape[2] * scale_factor), mode)

        def call(self, inputs):
            return self.upsample(inputs)

    class TFConcat(keras.layers.Layer):
        # TF версия torch.concat()
        def __init__(self, dimension=1, w=None):
            super().__init__()
            assert dimension == 1, "Только для NCHW → NHWC"
            self.d = 3

        def call(self, inputs):
            return tf.concat(inputs, self.d)

    def parse_model(d, ch, model, imgsz):  # model_dict, input_channels(3)
        LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
        anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # Количество якорных боксов
        no = na * (nc + 5)  # Количество выходов

        layers, save, c2 = [], [], ch[-1]  # Слои, сохранение, каналы выхода
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
            m_str = m
            m = eval(m) if isinstance(m, str) else m  # Преобразование строки в класс
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # Преобразование аргументов
                except NameError:
                    pass

            n = max(round(n * gd), 1) if n > 1 else n  # Глубина сети
            if m in [
                nn.Conv2d, Conv, DWConv, DWConvTranspose2d, Bottleneck, SPP, SPPF, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3x]:
                c1, c2 = ch[f], args[0]
                c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3x]:
                    args.insert(2, n)
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
            elif m in [Detect, Segment]:
                args.append([ch[x + 1] for x in f])
                if isinstance(args[1], int):
                    args[1] = [list(range(args[1] * 2))] * len(f)
                if m is Segment:
                    args[3] = make_divisible(args[3] * gw, 8)
                args.append(imgsz)
            else:
                c2 = ch[f]

            tf_m = eval('TF' + m_str.replace('nn.', ''))
            m_ = keras.Sequential([tf_m(*args, w=model.model[i][j]) for j in range(n)]) if n > 1 \
                else tf_m(*args, w=model.model[i])  # Модуль

            torch_m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # Модуль PyTorch
            t = str(m)[8:-2].replace('__main__.', '')  # Тип модуля
            np = sum(x.numel() for x in torch_m_.parameters())  # Количество параметров
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # Привязка метаданных
            LOGGER.info(f'{i:>3}{str(f):>18}{str(n):>3}{np:>10}  {t:<40}{str(args):<30}')  # Вывод информации
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # Сохранение индексов
            layers.append(m_)
            ch.append(c2)
        return keras.Sequential(layers), sorted(save)

    class TFModel:
        # TensorFlow модель YOLOv5
        def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, model=None, imgsz=(640, 640)):
            super().__init__()
            if isinstance(cfg, dict):
                self.yaml = cfg  # Словарь модели
            else:  # *.yaml
                import yaml
                self.yaml_file = Path(cfg).name
                with open(cfg) as f:
                    self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # Загрузка конфига

            # Определение модели
            if nc and nc != self.yaml['nc']:
                LOGGER.info(f"Переопределение {cfg} nc={self.yaml['nc']} → nc={nc}")
                self.yaml['nc'] = nc  # Переопределение количества классов
            self.model, self.savelist = parse_model(deepcopy(self.yaml), ch=[ch], model=model, imgsz=imgsz)

        def predict(self,
                    inputs,
                    tf_nms=False,
                    agnostic_nms=False,
                    topk_per_class=100,
                    topk_all=100,
                    iou_thres=0.45,
                    conf_thres=0.25):
            y = []  # Список для хранения выходных данных
            x = inputs  # Входные данные
            for m in self.model.layers:
                if m.f != -1:  # Если слой не использует предыдущий выход
                    # Извлечение данных из предыдущих слоев
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

                x = m(x)  # Прямой проход через текущий слой
                # Сохраняем выход, если слой указан в savelist
                y.append(x if m.i in self.savelist else None)

            # Применение TensorFlow NMS
            if tf_nms:
                # Преобразование координат из xywh в xyxy
                boxes = self._xywh2xyxy(x[0][..., :4])
                probs = x[0][:, :, 4:5]  # Уверенности
                classes = x[0][:, :, 5:]  # Классы
                scores = probs * classes  # Суммарные оценки
                if agnostic_nms:
                    # Безклассовая NMS
                    nms = AgnosticNMS()((boxes, classes, scores), topk_all, iou_thres, conf_thres)
                else:
                    # Классовая NMS
                    boxes = tf.expand_dims(boxes, 2)
                    nms = tf.image.combined_non_max_suppression(boxes,
                                                                scores,
                                                                topk_per_class,
                                                                topk_all,
                                                                iou_thres,
                                                                conf_thres,
                                                                clip_boxes=False)
                return (nms,)
            return x  # Выход в формате [1,6300,85] = [xywh, conf, class0, class1, ...]

        @staticmethod
        def _xywh2xyxy(xywh):
            # Преобразование координат из [x, y, w, h] в [x1, y1, x2, y2]
            x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
            return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)

    class AgnosticNMS(keras.layers.Layer):
        # Класс для безклассовой NMS в TensorFlow
        def call(self, input, topk_all, iou_thres, conf_thres):
            # Обертка map_fn для обработки батча
            return tf.map_fn(lambda x: self._nms(x, topk_all, iou_thres, conf_thres),
                             input,
                             fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.int32),
                             name='agnostic_nms')

        @staticmethod
        def _nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25):  # Безклассовая NMS
            boxes, classes, scores = x  # Входные данные: боксы, классы, оценки
            class_inds = tf.cast(tf.argmax(classes, axis=-1), tf.float32)  # Индексы классов
            scores_inp = tf.reduce_max(scores, -1)  # Максимальные оценки
            # Нахождение индексов выбранных боксов
            selected_inds = tf.image.non_max_suppression(boxes,
                                                         scores_inp,
                                                         max_output_size=topk_all,
                                                         iou_threshold=iou_thres,
                                                         score_threshold=conf_thres)
            # Выбор и заполнение результатов
            selected_boxes = tf.gather(boxes, selected_inds)
            padded_boxes = tf.pad(selected_boxes,
                                  paddings=[[0, topk_all - tf.shape(selected_boxes)[0]], [0, 0]],
                                  constant_values=0.0)
            selected_scores = tf.gather(scores_inp, selected_inds)
            padded_scores = tf.pad(selected_scores,
                                   paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
                                   constant_values=-1.0)
            selected_classes = tf.gather(class_inds, selected_inds)
            padded_classes = tf.pad(selected_classes,
                                    paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
                                    constant_values=-1.0)
            valid_detections = tf.shape(selected_inds)[0]  # Количество валидных детекций
            return padded_boxes, padded_scores, padded_classes, valid_detections


def activations(act=nn.SiLU):
    # Возвращает эквивалентную активацию TensorFlow для заданной активации PyTorch
    if isinstance(act, nn.LeakyReLU):
        # Возвращает функцию LeakyReLU для TensorFlow
        return lambda x: keras.activations.relu(x, alpha=0.1)
    elif isinstance(act, nn.Hardswish):
        # Возвращает функцию Hardswish для TensorFlow
        return lambda x: x * tf.nn.relu6(x + 3) * 0.166666667
    elif isinstance(act, (nn.SiLU, SiLU)):
        # Возвращает функцию Swish для TensorFlow
        return lambda x: keras.activations.swish(x)
    else:
        # Если нет соответствующей активации TensorFlow, вызывает исключение
        raise Exception(f'no matching TensorFlow activation found for PyTorch activation {act}')


def representative_dataset_gen(dataset, ncalib=100):
    # Генератор представительного набора данных для использования с converter.representative_dataset.
    # Возвращает генератор массивов NumPy.
    for n, (path, img, im0s, vid_cap, string) in enumerate(dataset):
        # Переставляем оси изображения из формата PyTorch (CHW) в формат TensorFlow (HWC)
        im = np.transpose(img, [1, 2, 0])
        # Добавляем размерность батча
        im = np.expand_dims(im, axis=0).astype(np.float32)
        # Нормализуем изображение
        im /= 255
        # Возвращаем нормализованное изображение
        yield [im]
        if n >= ncalib:
            # Прерываем цикл, если достигнуто заданное количество образцов для калибровки
            break


def run(
        weights=ROOT / 'yolov5s.pt',  # Путь к весам модели PyTorch
        imgsz=(640, 640),  # Размер изображения для инференса (высота, ширина)
        batch_size=1,  # Размер батча
        dynamic=False,  # Флаг динамического размера батча
):
    # PyTorch модель
    # Создаем тензор-изображение для входа в PyTorch модель (формат BCHW)
    im = torch.zeros((batch_size, 3, *imgsz))
    # Загружаем модель PyTorch с заданными весами
    model = attempt_load(weights, device=torch.device('cpu'), inplace=True, fuse=False)
    # Проводим инференс на пустом изображении
    _ = model(im)
    # Выводим информацию о модели
    model.info()

    # TensorFlow модель
    # Создаем тензор-изображение для входа в TensorFlow модель (формат BHWC)
    im = tf.zeros((batch_size, *imgsz, 3))
    # Создаем экземпляр TensorFlow модели
    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    # Проводим инференс на пустом изображении
    _ = tf_model.predict(im)

    # Keras модель
    # Создаем входной слой Keras модели
    im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
    # Создаем экземпляр Keras модели
    keras_model = keras.Model(inputs=im, outputs=tf_model.predict(im))
    # Выводим сводку о Keras модели
    keras_model.summary()

    # Выводим сообщение об успешной проверке моделей
    LOGGER.info('PyTorch, TensorFlow and Keras models successfully verified.\nUse export.py for TF model export.')


def parse_opt():
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser()
    # Добавляем аргумент для пути к весам модели
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='weights path')
    # Добавляем аргумент для размера изображения для инференса
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # Добавляем аргумент для размера батча
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    # Добавляем аргумент для динамического размера батча
    parser.add_argument('--dynamic', action='store_true', help='dynamic batch size')
    # Парсим аргументы командной строки
    opt = parser.parse_args()
    # Убеждаемся, что размер изображения задан в виде двух значений (высота, ширина)
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    # Выводим аргументы
    print_args(vars(opt))
    return opt


def main(opt):
    # Вызываем функцию run с распакованными аргументами
    run(**vars(opt))


if __name__ == "__main__":
    # Парсим аргументы командной строки
    opt = parse_opt()
    # Вызываем основную функцию
    main(opt)