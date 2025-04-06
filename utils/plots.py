# YOLOv5 🚀 от Ultralytics, лицензия GPL-3.0
"""
Утилиты для построения графиков
"""

import contextlib
import math
import os
from copy import copy
from pathlib import Path
from urllib.error import URLError

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

from utils import TryExcept, threaded
from utils.general import (CONFIG_DIR, FONT, LOGGER, check_font, check_requirements, clip_boxes, increment_path,
                           is_ascii, xywh2xyxy, xyxy2xywh)
from utils.metrics import fitness
from utils.segment.general import scale_image

# Настройки
RANK = int(os.getenv('RANK', -1))
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # только для записи в файлы


class Colors:
    # Палитра цветов Ultralytics https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # порядок rgb (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # создать экземпляр для 'from utils.plots import colors'


def check_pil_font(font=FONT, size=10):
    # Возвращает шрифт PIL TrueType, загружая его в CONFIG_DIR, если необходимо
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception:  # загрузить, если отсутствует
        try:
            check_font(font)
            return ImageFont.truetype(str(font), size)
        except TypeError:
            check_requirements('Pillow>=8.4.0')  # известная проблема https://github.com/ultralytics/yolov5/issues/5374
        except URLError:  # не в сети
            return ImageFont.load_default()


class Annotator:
    # Аннотатор YOLOv5 для мозаик train/val и аннотаций для обнаружения и вывода результатов
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Изображение не является непрерывным. Примените np.ascontiguousarray(im) к входному изображению в Annotator().'
        non_ascii = not is_ascii(example)  # не латинские метки, например, азиатские, арабские, кириллица
        self.pil = pil or non_ascii
        if self.pil:  # использовать PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font='Arial.Unicode.ttf' if non_ascii else font,
                                       size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # использовать cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # ширина линии

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Добавить один bounding box к изображению с меткой
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # bounding box
            if label:
                w, h = self.font.getsize(label)  # ширина и высота текста
                outside = box[1] - h >= 0  # метка помещается снаружи bounding box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            pass
           #p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            #cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            #if label:
                #tf = max(self.lw - 1, 1)  # толщина шрифта
                #w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # ширина и высота текста
                #outside = p1[1] - h >= 3
               # p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                #cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # заполнить
                #cv2.putText(self.im,
                           # label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            #0,
                            #self.lw / 3,
                            #txt_color,
                            #thickness=tf,
                            #lineType=cv2.LINE_AA)

    def masks(self, masks, colors, im_gpu=None, alpha=1):
        """Построение масок одновременно.
        Аргументы:
            masks (tensor): предсказанные маски на cuda, размер: [n, h, w]
            colors (List[List[Int]]): цвета для предсказанных масок, [[r, g, b] * n]
            im_gpu (tensor): изображение на cuda, размер: [3, h, w], диапазон: [0, 1]
            alpha (float): прозрачность маски: 0.0 полностью прозрачно, 1.0 непрозрачно
        """
        if self.pil:
            # преобразовать в numpy сначала
            self.im = np.asarray(self.im).copy()
        if im_gpu is None:
            # Добавить несколько масок размера (h,w,n) с цветами списка ([r,g,b], [r,g,b], ...)
            if len(masks) == 0:
                return
            if isinstance(masks, torch.Tensor):
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                masks = masks.permute(1, 2, 0).contiguous()
                masks = masks.cpu().numpy()
            # masks = np.ascontiguousarray(masks.transpose(1, 2, 0))
            masks = scale_image(masks.shape[:2], masks, self.im.shape)
            masks = np.asarray(masks, dtype=np.float32)
            colors = np.asarray(colors, dtype=np.float32)  # размер (n,3)
            s = masks.sum(2, keepdims=True).clip(0, 1)  # сложить все маски вместе
            masks = (masks @ colors).clip(0, 255)  # (h,w,n) @ (n,3) = (h,w,3)
            self.im[:] = masks * alpha + self.im * (1 - s * alpha)
        else:
            if len(masks) == 0:
                self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
            colors = torch.tensor(colors, device=im_gpu.device, dtype=torch.float32) / 255.0
            colors = colors[:, None, None]  # размер(n,1,1,3)
            masks = masks.unsqueeze(3)  # размер(n,h,w,1)
            masks_color = masks * (colors * alpha)  # размер(n,h,w,3)

            inv_alph_masks = (1 - masks * alpha).cumprod(0)  # размер(n,h,w,1)
            mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand размер(n,h,w,3)
            im_gpu = im_gpu.flip(dims=[0])  # flip channel
            im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # размер(h,w,3)
            im_gpu = inv_alph_masks[-1]  # только предсказанные результаты
            im_mask = (im_gpu * 255).byte().cpu().numpy()
            self.im[:] = scale_image(im_gpu.shape, im_mask, self.im.shape)
        if self.pil:
            # преобразовать im обратно в PIL и обновить draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Добавить прямоугольник к изображению (только для PIL)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top'):
        # Добавить текст к изображению (только для PIL)
        if anchor == 'bottom':  # начать y с нижней части шрифта
            w, h = self.font.getsize(text)  # ширина и высота текста
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        # Обновить self.im из массива numpy
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        # Вернуть аннотированное изображение в виде массива
        return np.asarray(self.im)


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    x:              Функции для визуализации
    module_type:    Тип модуля
    stage:          Этап модуля в модели
    n:              Максимальное количество карт признаков для построения графика
    save_dir:       Каталог для сохранения результатов
    """
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape  # размеры пакета, каналы, высота, ширина
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # имя файла

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # выбрать индекс пакета 0, блокировать по каналам
            n = min(n, channels)  # количество графиков
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 строк x n/8 столбцов
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            LOGGER.info(f'Сохранение {f}... ({n}/{channels})')
            plt.savefig(f, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # сохранить в формате npy


def hist2d(x, y, n=100):
    # 2D гистограмма, используемая в labels.png и evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    from scipy.signal import butter, filtfilt

    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # двунаправленный фильтр


def output_to_target(output, max_det=300):
    # Преобразовать выход модели в целевой формат [batch_id, class_id, x, y, w, h, conf] для построения графика
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, xyxy2xywh(box), conf), 1))
    return torch.cat(targets, 0).numpy()


@threaded
def plot_images(images, targets, paths=None, fname='images.jpg', names=None):
    # Построить сетку изображений с метками
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    max_size = 1920  # максимальный размер изображения
    max_subplots = 16  # максимальное количество подграфиков, например 4x4
    bs, _, h, w = images.shape  # размеры пакета, _, высота, ширина
    bs = min(bs, max_subplots)  # ограничить количество изображений для построения графика
    ns = np.ceil(bs ** 0.5)  # количество подграфиков (квадрат)
    if np.max(images[0]) <= 1:
        images *= 255  # денормализация (опционально)

    # Создать изображение
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # инициализация
    for i, im in enumerate(images):
        if i == max_subplots:  # если в последнем пакете меньше изображений, чем мы ожидаем
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # начало блока
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    # Изменить размер (опционально)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Аннотировать
    fs = int((h + w) * ns * 0.01)  # размер шрифта
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # начало блока
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # границы
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # имена файлов
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # цели изображения
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 6  # метки, если нет столбца conf
            conf = None if labels else ti[:, 6]  # проверить наличие уверенности (метка vs предсказание)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # если нормализовано с допуском 0.01
                    boxes[[0, 2]] *= w  # масштабировать до пикселей
                    boxes[[1, 3]] *= h
                elif scale < 1:  # абсолютные координаты требуют масштабирования, если изменяется размер изображения
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # порог уверенности 0.25
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
    annotator.im.save(fname)  # сохранить


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Построить график изменения Learning Rate, имитируя обучение на полных эпохах
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # не изменять оригиналы
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('эпоха')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def plot_val_txt():  # from utils.plots import *; plot_val()
    # Построить гистограммы val.txt
    x = np.loadtxt('val.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.plots import *; plot_targets_txt()
    # Построить гистограммы targets.txt
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x цели', 'y цели', 'ширина цели', 'высота цели']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label=f'{x[i].mean():.3g} +/- {x[i].std():.3g}')
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_val_study(file='', dir='', x=None):  # from utils.plots import *; plot_val_study()
    # Построить файл study.txt, сгенерированный val.py (или построить все study*.txt в каталоге)
    save_dir = Path(file).parent if file else Path(dir)
    plot2 = False  # построить дополнительные результаты
    if plot2:
        ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [save_dir / f'study_coco_{x}.txt' for x in ['yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6']]:
    for f in sorted(save_dir.glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        if plot2:
            s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_preprocess (ms/img)', 't_inference (ms/img)', 't_NMS (ms/img)']
            for i in range(7):
                ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
                ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[5, 1:j],
                 y[3, 1:j] * 1E2,
                 '.-',
                 linewidth=2,
                 markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-',
             linewidth=2,
             markersize=8,
             alpha=.25,
             label='EfficientDet')

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(25, 55)
    ax2.set_xlabel('Скорость GPU (ms/изображение)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    f = save_dir / 'study.png'
    print(f'Сохранение {f}...')
    plt.savefig(f, dpi=300)


@TryExcept()  # известная проблема https://github.com/ultralytics/yolov5/issues/5395
def plot_labels(labels, names=(), save_dir=Path('')):
    # Построить метки датасета
    LOGGER.info(f'Построение меток в {save_dir / 'labels.jpg'}... ')
    c, b = labels[:, 0], labels[:, 1:].transpose()  # классы, bounding boxes
    nc = int(c.max() + 1)  # количество классов
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'ширина', 'высота'])

    # seaborn коррелограмма
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib метки
    matplotlib.use('svg')  # быстрее
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    with contextlib.suppress(Exception):  # окрасить столбцы гистограммы по классам
        [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # известная проблема #3195
    ax[0].set_ylabel('экземпляры')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('классы')
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='ширина', y='высота', ax=ax[3], bins=50, pmax=0.9)

    # прямоугольники
    labels[:, 1:3] = 0.5  # центр
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # построить
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()


def imshow_cls(im, labels=None, pred=None, names=None, nmax=25, verbose=False, f=Path('images.jpg')):
    # Показать сетку изображений классификации с метками (опционально) и предсказаниями (опционально)
    from utils.augmentations import denormalize

    names = names or [f'class{i}' for i in range(1000)]
    blocks = torch.chunk(denormalize(im.clone()).cpu().float(), len(im),
                         dim=0)  # выбрать индекс пакета 0, блокировать по каналам
    n = min(len(blocks), nmax)  # количество графиков
    m = min(8, round(n ** 0.5))  # 8 x 8 по умолчанию
    fig, ax = plt.subplots(math.ceil(n / m), m)  # 8 строк x n/8 столбцов
    ax = ax.ravel() if m > 1 else [ax]
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax[i].imshow(blocks[i].squeeze().permute((1, 2, 0)).numpy().clip(0.0, 1.0))
        ax[i].axis('off')
        if labels is not None:
            s = names[labels[i]] + (f'—{names[pred[i]]}' if pred is not None else '')
            ax[i].set_title(s, fontsize=8, verticalalignment='top')
    plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        LOGGER.info(f'Сохранение {f}')
        if labels is not None:
            LOGGER.info('Истинные значения:     ' + ' '.join(f'{names[i]:3s}' for i in labels[:nmax]))
        if pred is not None:
            LOGGER.info('Предсказанные значения:' + ' '.join(f'{names[i]:3s}' for i in pred[:nmax]))
    return f


def plot_evolve(evolve_csv='path/to/evolve.csv'):  # from utils.plots import *; plot_evolve()
    # Построить результаты эволюции гиперпараметров из evolve.csv
    evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)  # индекс максимальной приспособленности
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    print(f'Лучшие результаты из строки {j} файла {evolve_csv}:')
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]  # лучший отдельный результат
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title(f'{k} = {mu:.3g}', fontdict={'size': 9})  # ограничить до 40 символов
        if i % 5 != 0:
            plt.yticks([])
        print(f'{k:>15}: {mu:.3g}')
    f = evolve_csv.with_suffix('.png')  # имя файла
    plt.savefig(f, dpi=200)
    plt.close()
    print(f'Сохранение {f}')


def plot_results(file='path/to/results.csv', dir=''):
    # Построить результаты обучения results.csv. Использование: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'Не найдено файлов results.csv в {save_dir.resolve()}, нечего строить.'
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j].astype('float')
                # y[y == 0] = np.nan  # не показывать нулевые значения
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)
                ax[i].set_title(s[j], fontsize=12)
                # if j in [8, 9, 10]:  # совместить оси y для потерь train и val
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.info(f'Предупреждение: ошибка построения графика для {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=200)
    plt.close()


def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    # Построить логи по изображениям iDetection '*.txt'. from utils.plots import *; profile_idetection()
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ['Изображения', 'Свободное хранилище (ГБ)', 'Использование ОЗУ (ГБ)', 'Батарея', 'dt_raw (мс)', 'dt_smooth (мс)', 'реальный FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]  # обрезать первые и последние строки
            n = results.shape[1]  # количество строк
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = (results[0] - results[0].min())  # установить t0=0с
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel('время (с)')
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print(f'Предупреждение: ошибка построения графика для {f}; {e}')
    ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)


def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Сохранить обрезанный фрагмент изображения как {file} с размером обрезки {gain} и {pad} пикселей. Сохранить и/или вернуть обрезку
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # bounding boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # попытаться преобразовать прямоугольник в квадрат
    b[:, 2:] = b[:, 2:] * gain + pad  # размер bounding box * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # создать каталог
        f = str(increment_path(file).with_suffix('.jpg'))
        # cv2.imwrite(f, crop)  # сохранить в формате BGR, https://github.com/ultralytics/yolov5/issues/7007 проблема с хроматическим субсэмплированием
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # сохранить в формате RGB
    return crop