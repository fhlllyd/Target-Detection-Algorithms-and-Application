import torch
import torch.nn as nn
import torch.nn.functional as F

from ..general import xywh2xyxy
from ..loss import FocalLoss, smooth_BCE
from ..metrics import bbox_iou
from ..torch_utils import de_parallel
from .general import crop_mask


class ComputeLoss:
    # Вычисление потерь
    def __init__(self, model, autobalance=False, overlap=False):
        self.sort_obj_iou = False
        self.overlap = overlap
        # Получение устройства модели
        device = next(model.parameters()).device
        h = model.hyp  # Гиперпараметры
        self.device = device

        # Определение критериев
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Сглаживание меток классов
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
        # Положительные и отрицательные цели BCE

        # Фокальная потеря
        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]
        # Модуль Detect()

        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        # Равномерность для P3-P7

        self.ssi = list(m.stride).index(16) if autobalance else 0
        # Индекс stride 16

        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # Количество якорных прямоугольников
        self.nc = m.nc  # Количество классов
        self.nl = m.nl  # Количество слоев
        self.nm = m.nm  # Количество масков
        self.anchors = m.anchors
        self.device = device

    def __call__(self, preds, targets, masks):
        # Предсказания, цели, модель
        p, proto = preds
        bs, nm, mask_h, mask_w = proto.shape
        # Размер батча, количество масков, высота и ширина маски

        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lseg = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors, tidxs, xywhn = self.build_targets(p, targets)
        # Цели

        # Потери
        for i, pi in enumerate(p):
            # Индекс слоя, предсказания слоя
            b, a, gj, gi = indices[i]
            # Изображение, якорь, координаты сетки

            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
            # Целевая.objectность

            n = b.shape[0]
            # Количество целей
            if n:
                pxy, pwh, _, pcls, pmask = pi[b, a, gj, gi].split((2, 2, 1, self.nc, nm), 1)
                # Подмножество предсказаний

                # Регрессия бокса
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                # Предсказанный бокс

                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                # IoU (предсказание, цель)
                lbox += (1.0 - iou).mean()
                # IoU потеря

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou
                # Коэффициент IoU

                # Классификация
                if self.nc > 1:
                    # Потеря класса (только если несколько классов)
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    # Цели
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)
                    # BCE

                # Регрессия маски
                if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                    # Сжатие
                    masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                marea = xywhn[i][:, 2:].prod(1)
                # Нормализованная ширина и высота маски

                mxyxy = xywh2xyxy(xywhn[i] * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device))
                for bi in b.unique():
                    j = b == bi
                    # Индекс совпадения
                    if self.overlap:
                        mask_gti = torch.where(masks[bi][None] == tidxs[i][j].view(-1, 1, 1), 1.0, 0.0)
                    else:
                        mask_gti = masks[tidxs[i]][j]
                    lseg += self.single_mask_loss(mask_gti, pmask[j], proto[bi], mxyxy[j], marea[j])

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            # Потеря objectness
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        lseg *= self.hyp["box"] / bs

        loss = lbox + lobj + lcls + lseg
        return loss * bs, torch.cat((lbox, lseg, lobj, lcls)).detach()

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # Потеря маски для одного изображения
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])
        # Предсказанная маска

        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()

    def build_targets(self, p, targets):
        # Создание целей для compute_loss(), входные цели (изображение, класс, x, y, w, h)
        na, nt = self.na, targets.shape[0]
        # Количество якорных прямоугольников, количество целей

        tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], []
        gain = torch.ones(8, device=self.device)
        # Нормализованный коэффициент для пространства сетки

        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        # Похоже на .repeat_interleave(nt)

        if self.overlap:
            batch = p[0].shape[0]
            ti = []
            for i in range(batch):
                num = (targets[:, 0] == i).sum()
                # Количество целей для каждого изображения
                ti.append(torch.arange(num, device=self.device).float().view(1, num).repeat(na, 1) + 1)
                # (na, num)
            ti = torch.cat(ti, 1)
            # (na, nt)
        else:
            ti = torch.arange(nt, device=self.device).float().view(1, nt).repeat(na, 1)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None], ti[..., None]), 2)
        # Добавление индексов якорных прямоугольников

        g = 0.5  # Смещение
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],
                # jk,jm,lk,lm
            ],
            device=self.device).float() * g
            # Смещения

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
            # Коэффициент xyxy

            # Соответствие целей якорным прямоугольникам
            t = targets * gain
            # shape(3,n,7)
            if nt:
                # Соответствия
                r = t[..., 4:6] / anchors[:, None]
                # Соотношение wh
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                # Сравнение
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']
                # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]
                # Фильтрация

                # Смещения
                gxy = t[:, 2:4]
                # Сеточные xy
                gxi = gain[[2, 3]] - gxy
                # Обратные
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Определение
            bc, gxy, gwh, at = t.chunk(4, 1)
            # (Изображение, класс), сеточные xy, сеточные wh, якорные прямоугольники

            (a, tidx), (b, c) = at.long().T, bc.long().T
            # Якорные прямоугольники, изображение, класс
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            # Сеточные индексы

            # Добавление
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            # Изображение, якорь, сетка
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            # Бокс
            anch.append(anchors[a])
            # Якорные прямоугольники
            tcls.append(c)
            # Класс
            tidxs.append(tidx)
            xywhn.append(torch.cat((gxy, gwh), 1) / gain[2:6])
            # Нормализованные xywh

        return tcls, tbox, indices, anch, tidxs, xywhn