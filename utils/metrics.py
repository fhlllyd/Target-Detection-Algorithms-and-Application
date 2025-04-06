# YOLOv5 🚀 от Ultralytics, лицензия GPL-3.0
"""
Метрики валидации модели
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import TryExcept, threaded


def fitness(x):
    # Оценка модели как взвешенная комбинация метрик
    w = [0.0, 0.0, 0.1, 0.9]  # веса для [точность, полнота, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    # Фильтрация с использованием бокового фильтра с долей f
    nf = round(len(y) * f * 2) // 2 + 1  # количество элементов фильтра (должно быть нечетным)
    p = np.ones(nf // 2)  # заполнение единицами
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # заполнение y
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # сглаженный y


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=""):
    """ Вычисление средней точности для каждого класса, учитывая кривые точности и полноты.
    Источник: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Аргументы
        tp:  истинные положительные результаты (nparray, nx1 или nx10).
        conf:  значение объектности от 0 до 1 (nparray).
        pred_cls:  предсказанные классы объектов (nparray).
        target_cls:  истинные классы объектов (nparray).
        plot:  построение кривой точности-полноты при mAP@0.5
        save_dir:  каталог для сохранения графика
    # Возвращает
        Средняя точность, вычисленная в py-faster-rcnn.
    """

    # Сортировка по объектности
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Поиск уникальных классов
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # количество классов, количество обнаружений

    # Создание кривой точности-полноты и вычисление AP для каждого класса
    px, py = np.linspace(0, 1, 1000), []  # для построения графика
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # количество меток
        n_p = i.sum()  # количество предсказаний
        if n_p == 0 or n_l == 0:
            continue

        # Накопление ложных и истинных положительных результатов
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Полнота
        recall = tpc / (n_l + eps)  # кривая полноты
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # отрицательный x, xp, так как xp уменьшается

        # Точность
        precision = tpc / (tpc + fpc)  # кривая точности
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # точность при значении полноты

        # AP из кривой полноты-точности
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # точность при mAP@0.5

    # Вычисление F1 (гармоническое среднее точности и полноты)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # список: только классы, имеющие данные
    names = dict(enumerate(names))  # в словарь
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Точность')
        plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Полнота')

    i = smooth(f1.mean(0), 0.1).argmax()  # индекс максимального F1
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # истинные положительные результаты
    fp = (tp / (p + eps) - tp).round()  # ложные положительные результаты
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """ Вычисление средней точности, учитывая кривые полноты и точности
    # Аргументы
        recall:    кривая полноты (список)
        precision: кривая точности (список)
    # Возвращает
        Средняя точность, кривая точности, кривая полноты
    """

    # Добавление sentinel значений в начало и конец
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Вычисление оболочки точности
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Интегрирование площади под кривой
    method = 'interp'  # методы: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-точечная интерполяция (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # интегрирование
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # точки, в которых изменяется ось x (полнота)
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # площадь под кривой

    return ap, mpre, mrec


class ConfusionMatrix:
    # Обновленная версия https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # количество классов
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Возвращает пересечение над объединением (индекс Джакарда) для bounding boxes.
        Ожидаются обе группы bounding boxes в формате (x1, y1, x2, y2).
        Аргументы:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Возвращает:
            None, обновляет матрицу ошибок соответствующим образом
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # фон FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # верно
            else:
                self.matrix[self.nc, gc] += 1  # истинный фон

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # предсказанный фон

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # истинные положительные результаты
        fp = self.matrix.sum(1) - tp  # ложные положительные результаты
        # fn = self.matrix.sum(0) - tp  # ложные отрицательные результаты (пропущенные обнаружения)
        return tp[:-1], fp[:-1]  # удалить класс фон

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    def plot(self, normalize=True, save_dir='', names=()):
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # нормализация столбцов
        array[array < 0.005] = np.nan  # не аннотировать (появится как 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # количество классов, имена
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # для размера меток
        labels = (0 < nn < 99) and (nn == nc)  # применить имена к меткам осей
        ticklabels = (names + ['background']) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # подавить предупреждение о пустой матрице
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           "size": 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_ylabel('Истинный')
        ax.set_ylabel('Предсказанный')
        ax.set_title('Матрица ошибок')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close(fig)

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Возвращает пересечение над объединением (IoU) для box1(1,4) и box2(n,4)

    # Получение координат bounding boxes
    if xywh:  # преобразование из xywh в xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Площадь пересечения
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Площадь объединения
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # ширина выпуклой оболочки (наименьший охватывающий прямоугольник)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # высота выпуклой оболочки
        if CIoU or DIoU:  # Расстояние или полная IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # квадрат диагонали выпуклой оболочки
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # квадрат расстояния между центрами
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # полная IoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # площадь выпуклой оболочки
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Возвращает пересечение над объединением (Jaccard index) для bounding boxes.
    Обе группы bounding boxes должны быть в формате (x1, y1, x2, y2).
    Аргументы:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Возвращает:
        iou (Tensor[N, M]): матрица NxM, содержащая значения IoU для каждой пары
            bounding boxes из box1 и box2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_ioa(box1, box2, eps=1e-7):
    """ Возвращает пересечение над площадью box2 для box1 и box2. Bounding boxes в формате x1y1x2y2
    box1:       np.array размера (4)
    box2:       np.array размера (nx4)
    возвращает: np.array размера (n)
    """

    # Получение координат bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Площадь пересечения
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # Площадь box2
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Пересечение над площадью box2
    return inter_area / box2_area


def wh_iou(wh1, wh2, eps=1e-7):
    # Возвращает матрицу nxm IoU. wh1 размера nx2, wh2 размера mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


# Графики ----------------------------------------------------------------------------------------------------------------


@threaded
def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Кривая точности-полноты
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # отображение легенды для каждого класса, если их меньше 21
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='все классы %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Полнота')
    ax.set_ylabel('Точность')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title('Кривая точности-полноты')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Уверенность', ylabel='Метрика'):
    # Кривая зависимости метрики от уверенности
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # отображение легенды для каждого класса, если их меньше 21
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'все классы {y.max():.2f} при {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f'Кривая зависимости {ylabel} от уверенности')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)