# YOLOv5 🚀 от Ultralytics, лицензия GPL-3.0
"""
Функции потерь
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # Возвращает целевые значения для позитивных и негативных примеров с сглаживанием меток
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEWithLogitLoss() с уменьшенным эффектом отсутствующих меток.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # должно быть nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        # Вычисление потерь
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # вероятность из логитов
        dx = pred - true  # уменьшение только эффекта отсутствующих меток
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Обертка для фокусной потери вокруг существующей функции потерь loss_fcn
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # должно быть nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # необходимо для применения FL к каждому элементу

    def forward(self, pred, true):
        # Вычисление потерь
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # вероятность из логитов
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Обертка для качества фокусной потери вокруг существующей функции потерь loss_fcn
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # должно быть nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # необходимо для применения FL к каждому элементу

    def forward(self, pred, true):
        # Вычисление потерь
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # вероятность из логитов
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Вычисление потерь
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # получить устройство модели
        h = model.hyp  # гиперпараметры

        # Определение критериев
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Сглаживание меток классов https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # целевые значения для позитивных и негативных примеров

        # Фокусная потеря
        g = h['fl_gamma']  # гамма для фокусной потери
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # модуль Detect()
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # баланс для уровней P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # индекс для шага 16
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # количество якорей
        self.nc = m.nc  # количество классов
        self.nl = m.nl  # количество уровней
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # предсказания, целевые значения
        lcls = torch.zeros(1, device=self.device)  # потери классов
        lbox = torch.zeros(1, device=self.device)  # потери bounding box
        lobj = torch.zeros(1, device=self.device)  # потери объектности
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # целевые значения

        # Вычисление потерь
        for i, pi in enumerate(p):  # индекс уровня, предсказания на уровне
            b, a, gj, gi = indices[i]  # индексы изображения, якоря, сетки y, сетки x
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # целевые объекты

            n = b.shape[0]  # количество целевых объектов
            if n:
                # Разделение предсказаний на координаты и классы
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # координаты и классы

                # Регрессия bounding box
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # предсказанный bounding box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # вычисление IoU
                lbox += (1.0 - iou).mean()  # потери IoU

                # Потери объектности
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # присвоение IoU

                # Потери классов
                if self.nc > 1:  # если более одного класса
                    t = torch.full_like(pcls, self.cn, device=self.device)  # целевые значения
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # потери BCE

            obji = self.BCEobj