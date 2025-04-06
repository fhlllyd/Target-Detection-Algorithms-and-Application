# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import numpy as np

from ..metrics import ap_per_class


def fitness(x):
    # Фитнес модели как взвешенная комбинация метрик
    w = [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.1, 0.9]
    return (x[:, :8] * w).sum(1)


def ap_per_class_box_and_mask(
        tp_m,
        tp_b,
        conf,
        pred_cls,
        target_cls,
        plot=False,
        save_dir=".",
        names=(),
):
    """
    Аргументы:
        tp_b: tp для боксов.
        tp_m: tp для масков.
        другие аргументы см. в `func: ap_per_class`.
    """
    results_boxes = ap_per_class(tp_b,
                                 conf,
                                 pred_cls,
                                 target_cls,
                                 plot=plot,
                                 save_dir=save_dir,
                                 names=names,
                                 prefix="Box")[2:]
    results_masks = ap_per_class(tp_m,
                                 conf,
                                 pred_cls,
                                 target_cls,
                                 plot=plot,
                                 save_dir=save_dir,
                                 names=names,
                                 prefix="Mask")[2:]

    results = {
        "boxes": {
            "p": results_boxes[0],
            "r": results_boxes[1],
            "ap": results_boxes[3],
            "f1": results_boxes[2],
            "ap_class": results_boxes[4]},
        "masks": {
            "p": results_masks[0],
            "r": results_masks[1],
            "ap": results_masks[3],
            "f1": results_masks[2],
            "ap_class": results_masks[4]}}
    return results


class Metric:

    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )

    @property
    def ap50(self):
        """AP@0.5 для всех классов.
        Возвращает:
            (nc, ) или [].
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """AP@0.5:0.95
        Возвращает:
            (nc, ) или [].
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """Средняя точность для всех классов.
        Возвращает:
            float.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """Среднее полнота для всех классов.
        Возвращает:
            float.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """Среднее AP@0.5 для всех классов.
        Возвращает:
            float.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """Среднее AP@0.5:0.95 для всех классов.
        Возвращает:
            float.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Среднее значений результатов, возвращает mp, mr, map50, map"""
        return (self.mp, self.mr, self.map50, self.map)

    def class_result(self, i):
        """Результат для конкретного класса, возвращает p[i], r[i], ap50[i], ap[i]"""
        return (self.p[i], self.r[i], self.ap50[i], self.ap[i])

    def get_maps(self, nc):
        maps = np.zeros(nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def update(self, results):
        """
        Аргументы:
            results: кортеж (p, r, ap, f1, ap_class)
        """
        p, r, all_ap, f1, ap_class_index = results
        self.p = p
        self.r = r
        self.all_ap = all_ap
        self.f1 = f1
        self.ap_class_index = ap_class_index


class Metrics:
    """Метрика для боксов и масков."""

    def __init__(self) -> None:
        self.metric_box = Metric()
        self.metric_mask = Metric()

    def update(self, results):
        """
        Аргументы:
            results: Словарь {'boxes': Словарь{}, 'masks': Словарь{}}
        """
        self.metric_box.update(list(results["boxes"].values()))
        self.metric_mask.update(list(results["masks"].values()))

    def mean_results(self):
        return self.metric_box.mean_results() + self.metric_mask.mean_results()

    def class_result(self, i):
        return self.metric_box.class_result(i) + self.metric_mask.class_result(i)

    def get_maps(self, nc):
        return self.metric_box.get_maps(nc) + self.metric_mask.get_maps(nc)

    @property
    def ap_class_index(self):
        # боксы и маски имеют один и тот же ap_class_index
        return self.metric_box.ap_class_index


KEYS = [
    "train/box_loss",
    "train/seg_loss",  # train loss
    "train/obj_loss",
    "train/cls_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP_0.5(B)",
    "metrics/mAP_0.5:0.95(B)",  # metrics
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP_0.5(M)",
    "metrics/mAP_0.5:0.95(M)",  # metrics
    "val/box_loss",
    "val/seg_loss",  # val loss
    "val/obj_loss",
    "val/cls_loss",
    "x/lr0",
    "x/lr1",
    "x/lr2",]

BEST_KEYS = [
    "best/epoch",
    "best/precision(B)",
    "best/recall(B)",
    "best/mAP_0.5(B)",
    "best/mAP_0.5:0.95(B)",
    "best/precision(M)",
    "best/recall(M)",
    "best/mAP_0.5(M)",
    "best/mAP_0.5:0.95(M)",]
