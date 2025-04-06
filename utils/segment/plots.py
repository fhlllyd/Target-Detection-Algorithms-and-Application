import contextlib
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .. import threaded
from ..general import xywh2xyxy
from ..plots import Annotator, colors


@threaded
def plot_images_and_masks(images, targets, masks, paths=None, fname='images.jpg', names=None):
    # Построение сетки изображений с метками
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)

    max_size = 1920  # Максимальный размер изображения
    max_subplots = 16  # Максимальное количество подграфиков, т.е. 4x4
    bs, _, h, w = images.shape  # Размер батча, _, высота, ширина
    bs = min(bs, max_subplots)  # Ограничение количества изображений для построения
    ns = np.ceil(bs ** 0.5)  # Количество подграфиков (квадрат)
    if np.max(images[0]) <= 1:
        images *= 255  # Де-нормализация (необязательно)

    # Создание изображения
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # Инициализация
    for i, im in enumerate(images):
        if i == max_subplots:  # Если в последнем батче меньше изображений, чем ожидается
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # П/poчало блока
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    # Масштабирование (необязательно)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Аннотация
    fs = int((h + w) * ns * 0.01)  # Размер шрифта
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # П/poчало блока
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # Границы
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # Имена файлов
        if len(targets) > 0:
            idx = targets[:, 0] == i
            ti = targets[idx]  # Цели для изображения

            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 6  # Метки, если нет столбца с доверием
            conf = None if labels else ti[:, 6]  # Проверка наличия доверия (метка vs предсказание)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # Если нормализованы с толерансом 0.01
                    boxes[[0, 2]] *= w  # Масштабирование до пикселей
                    boxes[[1, 3]] *= h
                elif scale < 1:  # Абсолютные координаты нуждаются в масштабировании, если изображение масштабируется
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # П/poгreshность доверия 0.25
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)

            # Построение масков
            if len(masks):
                if masks.max() > 1.0:  # Это означает, что маски перекрываются
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = len(ti)
                    index = np.arange(nl).reshape(nl, 1, 1) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)
                else:
                    image_masks = masks[idx]

                im = np.asarray(annotator.im).copy()
                for j, box in enumerate(boxes.T.tolist()):
                    if labels or conf[j] > 0.25:  # П/poгreshность доверия 0.25
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            im[y:y + h, x:x + w, :][mask] = im[y:y + h, x:x + w, :][mask] * 0.4 + np.array(color) * 0.6
                annotator.fromarray(im)
    annotator.im.save(fname)  # Сохранение


def plot_results_with_masks(file="path/to/results.csv", dir="", best=True):
    # Построение результатов обучения из results.csv. Использование: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            index = np.argmax(0.9 * data.values[:, 8] + 0.1 * data.values[:, 7] + 0.9 * data.values[:, 12] +
                              0.1 * data.values[:, 11])
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]):
                y = data.values[:, j]
                # y[y == 0] = np.nan  # Не показывать нулевые значения
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=2)
                if best:
                    # Лучший результат
                    ax[i].scatter(index, y[index], color="r", label=f"best:{index}", marker="*", linewidth=3)
                    ax[i].set_title(s[j] + f"\n{round(y[index], 5)}")
                else:
                    # Последний результат
                    ax[i].scatter(x[-1], y[-1], color="r", label="last", marker="*", linewidth=3)
                    ax[i].set_title(s[j] + f"\n{round(y[-1], 5)}")
                # if j in [8, 9, 10]:  # Общий ось y для потерь обучения и валидации
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print(f"Warning: Plotting error for {f}: {e}")
    ax[1].legend()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close()