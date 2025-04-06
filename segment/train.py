# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 segment model on a segment dataset
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640  # from pretrained (recommended)
    $ python segment/train.py --data coco128-seg.yaml --weights '' --cfg yolov5s-seg.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --data coco128-seg.yaml --weights yolov5s-seg.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Корневая директория YOLOv5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Добавляем в системный путь
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Относительный путь

import segment.val as validate  # Валидация модели
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, one_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from utils.loggers import GenericLogger
from utils.plots import plot_evolve, plot_labels
from utils.segment.dataloaders import create_dataloader
from utils.segment.loss import ComputeLoss
from utils.segment.metrics import KEYS, fitness
from utils.segment.plots import plot_images_and_masks, plot_results_with_masks
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # Ранг локального процесса
RANK = int(os.getenv('RANK', -1))  # Ранг процесса в分布式 обучении
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # Количество процессов
GIT_INFO = check_git_info()  # Информация о гит-репозитории


def train(hyp, opt, device, callbacks):  # hyp - путь к файлу гиперпараметров или словарь
    # Распаковка параметров
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, mask_ratio = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
            opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.mask_ratio

    # Папка для сохранения весов
    w = save_dir / 'weights'
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # Создание директории
    last, best = w / 'last.pt', w / 'best.pt'  # Путь к последнему и лучшему чекпоинту

    # Загрузка гиперпараметров
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # Загрузка из YAML
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # Сохранение гиперпараметров в opt

    # Сохранение настроек
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Инициализация логгеров
    data_dict = None
    if RANK in {-1, 0}:
        logger = GenericLogger(opt=opt, console_logger=LOGGER)

    # Конфигурация датасета
    plots = not evolve and not opt.noplots  # Включение графиков
    overlap = not opt.no_overlap  # Включение перекрытия для валидации
    cuda = device.type != 'cpu'  # Проверка наличия CUDA
    init_seeds(opt.seed + 1 + RANK, deterministic=True)  # Инициализация种子ов
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # Проверка датасета
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # Количество классов
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # Имена классов
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # Проверка COCO датасета

    # Загрузка модели
    check_suffix(weights, '.pt')  # Проверка расширения
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # Загрузка весов
        ckpt = torch.load(weights, map_location='cpu')  # Загрузка чекпоинта
        model = SegmentationModel(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # Исключаемые слои
        csd = ckpt['model'].float().state_dict()  # Состояние модели
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # Пересечение слоев
        model.load_state_dict(csd, strict=False)  # Загрузка весов
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')
    else:
        model = SegmentationModel(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # Создание новой модели

    # Заморозка слоев
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # Слои для заморозки
    for k, v in model.named_parameters():
        v.requires_grad = True  # Разрешение обучения
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False  # Заморозка

    # Размер изображения
    gs = max(int(model.stride.max()), 32)  # Размер сетки (максимальный шаг)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # Проверка размера изображения

    # Размер батча
    if RANK == -1 and batch_size == -1:  # Оценивание оптимального батча
        batch_size = check_train_batch_size(model, imgsz, amp)
        logger.update_params({"batch_size": batch_size})

    # Оптимизатор
    nbs = 64  # Номинальный размер батча
    accumulate = max(round(nbs / batch_size), 1)  # Аккумуляция градиентов
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # Масштаб веса регуляризации
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler (Расписание изменения скорости обучения)
    if opt.cos_lr:
        # Функция изменения скорости обучения по косинусному закону от 1 до hyp['lrf'] за весь период обучения
        lf = one_cycle(1, hyp['lrf'], epochs)
    else:
        # Линейная функция изменения скорости обучения
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    # Создание планировщика скорости обучения
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA (Экспоненциальное скользящее среднее модели)
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume (Восстановление обучения с сохраненной точки)
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            # Восстановление параметров обучения
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode (Режим Data Parallel, обучение на нескольких GPU)
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm (Синхронизация BatchNorm между GPU)
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader (Загрузчик тренировочных данных)
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == 'val' else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr('train: '),
        shuffle=True,
        mask_downsample_ratio=mask_ratio,
        overlap_mask=overlap,
    )
    # Получение меток всех объектов в тренировочном наборе
    labels = np.concatenate(dataset.labels, 0)
    # Максимальный номер класса в метках
    mlc = int(labels[:, 0].max())
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0 (Процесс с рангом 0, обычно используется для валидации и логирования)
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       mask_downsample_ratio=mask_ratio,
                                       overlap_mask=overlap,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            if not opt.noautoanchor:
                # Проверка и корректировка якорных ящиков
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # Предварительное снижение точности якорных ящиков

            if plots:
                # Построение графика распределения меток классов
                plot_labels(labels, names, save_dir)
        # callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode (Режим Distributed Data Parallel, эффективное обучение на нескольких GPU)
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes (Атрибуты модели)
    # Количество слоев детекции
    nl = de_parallel(model).model[-1].nl
    # Масштабирование коэффициента потерь для рамок
    hyp['box'] *= 3 / nl
    # Масштабирование коэффициента потерь для классов
    hyp['cls'] *= nc / 80 * 3 / nl
    # Масштабирование коэффициента потерь для объектов
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # Присоединение количества классов к модели
    model.hyp = hyp  # Присоединение гиперпараметров к модели
    # Присоединение весов классов к модели
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # Start training (Начало обучения)
    t0 = time.time()
    nb = len(train_loader)  # Количество батчей в тренировочном наборе
    # Количество итераций разогрева
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    last_opt_step = -1
    maps = np.zeros(nc)  # Средняя точность по классам (mAP)
    # Результаты: точность, полнота, mAP@.5, mAP@.5-.95, потери на валидации (рамки, объекты, классы)
    results = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # Не менять
    # Скейлер для автоматической смешанной точности
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # Класс для ранней остановки обучения
    stopper, stop = EarlyStopping(patience=opt.patience), False
    # Инициализация функции потерь
    compute_loss = ComputeLoss(model, overlap=overlap)
    # callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        # callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (Обновление весов изображений, опционально, только для одного GPU)
        if opt.image_weights:
            # Веса классов, учитывающие текущую точность по классам
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            # Веса изображений на основе весов классов
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
            # Случайный выбор индексов изображений с учетом весов
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)

        # Update mosaic border (Обновление границ мозаики, опционально)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # Средние потери
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 8) %
                    ('Epoch', 'GPU_mem', 'box_loss', 'seg_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # Прогресс-бар
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _, masks) in pbar:  # batch ------------------------------------------------------
            # callbacks.run('on_train_batch_start')
            # Количество интегрированных батчей с начала обучения
            ni = i + nb * epoch
            # Перемещение изображений на устройство и нормализация
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # Warmup (Разогрев обучения)
            if ni <= nw:
                xi = [0, nw]  # Интерполяция по итерациям
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # Соотношение потерь IoU (obj_loss = 1.0 или iou)
                # Накопление градиентов
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # Скорость обучения для смещения снижается от 0.1 до lr0, для остальных параметров - от 0.0 до lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        # Моментум изменяется от warmup_momentum до momentum
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
                        # Multi-scale training (Мульти-скейл обучение)
                        if opt.multi_scale:
                            # Генерация случайного размера изображения в пределах [0.5*imgsz, 1.5*imgsz]
                            sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                            # Коэффициент масштабирования
                            sf = sz / max(imgs.shape[2:])
                            if sf != 1:
                                # Новый размер с округлением до кратного gs
                                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                                # Интерполяция изображения в новый размер
                                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                        # Forward pass (Прямой проход)
                        with torch.cuda.amp.autocast(amp):
                            pred = model(imgs)  # Получение предсказаний
                            # Вычисление потерь
                            loss, loss_items = compute_loss(pred, targets.to(device), masks=masks.to(device).float())
                            # Корректировка потерь для DDP
                            if RANK != -1:
                                loss *= WORLD_SIZE
                            # Корректировка потерь для quad augmentation
                            if opt.quad:
                                loss *= 4.

                        # Backward pass (Обратный проход)
                        scaler.scale(loss).backward()

                        # Оптимизация
                        if ni - last_opt_step >= accumulate:
                            scaler.unscale_(optimizer)  # Убираем масштаб для градиентов
                            # Обрезка градиентов
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                            scaler.step(optimizer)  # Шаг оптимизатора
                            scaler.update()
                            optimizer.zero_grad()
                            # Обновление EMA модели
                            if ema:
                                ema.update(model)
                            last_opt_step = ni

                        # Логирование
                        if RANK in {-1, 0}:
                            # Средние потери
                            mloss = (mloss * i + loss_items) / (i + 1)
                            # Использование памяти GPU
                            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                            # Обновление прогресс-бара
                            pbar.set_description(('%11s' * 2 + '%11.4g' * 6) %
                                                 (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0],
                                                  imgs.shape[-1]))
                            # callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths)

                            # Визуализация мозаик
                            if plots:
                                if ni < 3:
                                    plot_images_and_masks(imgs, targets, masks, paths,
                                                          save_dir / f"train_batch{ni}.jpg")
                                if ni == 10:
                                    files = sorted(save_dir.glob('train*.jpg'))
                                    logger.log_images(files, "Mosaics", epoch)
                        # end batch ------------------------------------------------------------------------------------------------

                    # Scheduler (Планировщик LR)
                    lr = [x['lr'] for x in optimizer.param_groups]
                    scheduler.step()

                    if RANK in {-1, 0}:
                        # mAP calculation (Расчет метрик)
                        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
                        if not noval or final_epoch:
                            results, maps, _ = validate.run(
                                data_dict,
                                batch_size=batch_size // WORLD_SIZE * 2,
                                imgsz=imgsz,
                                half=amp,
                                model=ema.ema,
                                single_cls=single_cls,
                                dataloader=val_loader,
                                save_dir=save_dir,
                                plots=False,
                                callbacks=callbacks,
                                compute_loss=compute_loss,
                                mask_downsample_ratio=mask_ratio,
                                overlap=overlap
                            )

                        # Fitness calculation (Критерий остановки)
                        fi = fitness(np.array(results).reshape(1, -1))
                        stop = stopper(epoch=epoch, fitness=fi)
                        if fi > best_fitness:
                            best_fitness = fi

                        # Логирование метрик
                        log_vals = list(mloss) + list(results) + lr
                        metrics_dict = dict(zip(KEYS, log_vals))
                        logger.log_metrics(metrics_dict, epoch)

                        # Сохранение модели
                        if (not nosave) or (final_epoch and not evolve):
                            ckpt = {
                                'epoch': epoch,
                                'best_fitness': best_fitness,
                                'model': deepcopy(de_parallel(model)).half(),
                                'ema': deepcopy(ema.ema).half(),
                                'updates': ema.updates,
                                'optimizer': optimizer.state_dict(),
                                'opt': vars(opt),
                                'git': GIT_INFO,
                                'date': datetime.now().isoformat()
                            }

                            # Сохранение чекпоинтов
                            torch.save(ckpt, last)
                            if best_fitness == fi:
                                torch.save(ckpt, best)
                            if opt.save_period > 0 and epoch % opt.save_period == 0:
                                torch.save(ckpt, w / f'epoch{epoch}.pt')
                                logger.log_model(w / f'epoch{epoch}.pt')
                            del ckpt

                    # Early stopping (Ранняя остановка)
                    if RANK != -1:
                        broadcast_list = [stop if RANK == 0 else None]
                        dist.broadcast_object_list(broadcast_list, 0)
                        if RANK != 0:
                            stop = broadcast_list[0]
                    if stop:
                        break  # Остановка всех процессов

    if RANK in {-1, 0}:
        # Вывод информации о завершении обучения
        LOGGER.info(f'\n{epoch - start_epoch + 1} эпох выполнено за {(time.time() - t0) / 3600:.3f} часов.')

        # Обработка чекпоинтов
        for f in last, best:
            if f.exists():
                # Удаление оптимизатора из чекпоинта для уменьшения размера файла
                strip_optimizer(f)

                # Валидация лучшей модели
                if f is best:
                    LOGGER.info(f'\nПроверка лучшей модели {f}...')
                    results, _, _ = validate.run(
                        data_dict,  # Конфигурация датасета
                        batch_size=batch_size // WORLD_SIZE * 2,  # Размер батча для валидации
                        imgsz=imgsz,  # Размер изображений
                        model=attempt_load(f, device).half(),  # Загрузка модели в полупrecision
                        iou_thres=0.65 if is_coco else 0.60,  # Порог IoU (оптимизированный для COCO)
                        single_cls=single_cls,  # Одно-/многоклассовая валидация
                        dataloader=val_loader,  # Загрузчик валидационных данных
                        save_dir=save_dir,  # Директория сохранения
                        save_json=is_coco,  # Сохранение в формате COCO
                        verbose=True,  # Подробный вывод
                        plots=plots,  # Включение визуализации
                        callbacks=callbacks,  # Коллбэки
                        compute_loss=compute_loss,  # Функция потерь
                        mask_downsample_ratio=mask_ratio,  # Коэффициент уменьшения маски
                        overlap=overlap  # Включение перекрытия
                    )

                    # Логирование метрик для COCO датасета
                    if is_coco:
                        metrics_dict = dict(zip(KEYS, list(mloss) + list(results) + lr))
                        logger.log_metrics(metrics_dict, epoch)

        # Логирование окончательных результатов
        logger.log_metrics(dict(zip(KEYS[4:16], results)), epochs)
        if not opt.evolve:
            logger.log_model(best, epoch)  # Логирование модели в TensorBoard

        # Генерация и сохранение графиков
        if plots:
            plot_results_with_masks(file=save_dir / 'results.csv')  # Создание сводного графика
            files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
            files = [(save_dir / f) for f in files if (save_dir / f).exists()]  # Фильтрация существующих файлов

            LOGGER.info(f"Результаты сохранены в {colorstr('bold', save_dir)}")
            logger.log_images(files, "Results", epoch + 1)  # Логирование изображений в TensorBoard
            logger.log_images(sorted(save_dir.glob('val*.jpg')), "Validation", epoch + 1)

    # Освобождение памяти GPU
    torch.cuda.empty_cache()
    return results  # Возвращение метрик качества

def parse_opt(known=False):
    # Создаем объект парсера аргументов
    parser = argparse.ArgumentParser()
    # Путь к начальным весам модели
    parser.add_argument('--weights', type=str, default=ROOT / '', help='initial weights path')
    # Путь к конфигурационному файлу модели (model.yaml)
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # Путь к конфигурационному файлу датасета (dataset.yaml)
    parser.add_argument('--data', type=str, default=ROOT / 'data/wheel-seg.yaml', help='dataset.yaml path')
    # Путь к файлу гиперпараметров
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    # Общее количество эпох обучения
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    # Размер батча для всех GPU, -1 для автоматического определения
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    # Размер изображений для обучения и валидации (пиксели)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # Флаг прямоугольного обучения
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # Флаг возобновления последнего обучения
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # Флаг сохранения только финального чекпоинта
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # Флаг валидации только на финальной эпохе
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # Флаг отключения AutoAnchor
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    # Флаг отключения сохранения графиков
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    # Количество генераций для эволюции гиперпараметров
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # Бакет Google Cloud Storage
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # Кэширование изображений (в RAM или на диск)
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    # Флаг использования взвешенного выбора изображений для обучения
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # Устройство CUDA (например, 0 или 0,1,2,3 или cpu)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # Флаг мульти scales обучения (изменение размера изображения +/- 50%)
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # Флаг обучения мультиклассовых данных как одноклассовых
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # Тип оптимизатора (SGD, Adam, AdamW)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    # Флаг использования SyncBatchNorm (доступно только в DDP режиме)
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # Максимальное количество воркеров dataloader (на каждый RANK в DDP режиме)
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    # Директория сохранения результатов обучения
    parser.add_argument('--project', default=ROOT / 'runs/train-seg', help='save to project/name')
    # Имя поддиректории для сохранения результатов
    parser.add_argument('--name', default='exp', help='save to project/name')
    # Флаг разрешения существующей_project/name без инкремента
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # Флаг启用 четырех-поточного dataloader
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # Флаг启用 косинусной scheduler LR
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    # Эпсилон сглаживания меток
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # П忍耐 EarlyStopping (количество эпох без улучшения)
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    # Ф.layers для заморозки (например, backbone=10, first3=0 1 2)
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    # Периодичность сохранения чекпоинта (отключить, если < 1)
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    # Глобальный seed для обучения
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    # Аргумент для автоматического DDP Multi-GPU (не модифицировать)
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Аргументы Instance Segmentation
    # Коэффициент downsample маски для сэконом памяти
    parser.add_argument('--mask-ratio', type=int, default=4, help='Downsample the truth masks to saving memory')
    # Флаг отключения перекрытия маск (обучение быстрее, но с稍低 mAP)
    parser.add_argument('--no-overlap', action='store_true', help='Overlap masks train faster at slightly less mAP')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    # Проверки
    if RANK in {-1, 0}:
        # Вывод аргументов
        print_args(vars(opt))
        # Проверка статуса Git
        check_git_status()
        # Проверка_requirements
        check_requirements()

    # Возобновление обучения
    if opt.resume and not opt.evolve:  # Возобновление из указанного или последнего last.pt
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # Train options yaml
        opt_data = opt.data  # Original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # Замена
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # Восстановление
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # Избежание HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # Проверки
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # Если стандартное имя проекта, переименовать в runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # Перенос resume в exist_ok и отключение resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # Использование model.yaml в качестве имени
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

        # Режим DDP
        device = select_device(opt.device, batch_size=opt.batch_size)
        # Если локальный ранг не равен -1, то выполняем следующие проверки и настройки
        if LOCAL_RANK != -1:
            # Сообщение об ошибке, если параметры несовместимы с мульти - GPU DDP - обучением YOLOv5
            msg = 'несовместимо с много GPU DDP - обучением YOLOv5'
            # Проверка, чтобы параметр image_weights не был установлен
            assert not opt.image_weights, f'--image-weights {msg}'
            # Проверка, чтобы параметр evolve не был установлен
            assert not opt.evolve, f'--evolve {msg}'
            # Проверка, чтобы параметр batch_size не был равен -1
            assert opt.batch_size != -1, f'AutoBatch с --batch-size -1 {msg}, пожалуйста, укажите корректный --batch-size'
            # Проверка, чтобы batch_size был кратным WORLD_SIZE
            assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} должен быть кратным WORLD_SIZE'
            # Проверка, чтобы количество CUDA - устройств было больше локального ранга
            assert torch.cuda.device_count() > LOCAL_RANK, 'недостаточно CUDA - устройств для DDP - команды'
            # Установка текущего CUDA - устройства
            torch.cuda.set_device(LOCAL_RANK)
            # Создание объекта устройства для текущего CUDA - устройства
            device = torch.device('cuda', LOCAL_RANK)
            # Инициализация группы процессов для распределенного обучения
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

        # Обучение
        if not opt.evolve:
            # Запуск обучения с заданными гиперпараметрами, параметрами и устройством
            train(opt.hyp, opt, device, callbacks)

        # Эволюция гиперпараметров (необязательно)
        else:
            # Метаданные для эволюции гиперпараметров (масштаб мутации 0 - 1, нижний предел, верхний предел)
            meta = {
                'lr0': (1, 1e-5, 1e-1),  # начальная скорость обучения (SGD = 1E - 2, Adam = 1E - 3)
                'lrf': (1, 0.01, 1.0),  # конечная скорость обучения OneCycleLR (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # моментум SGD/бета1 Adam
                'weight_decay': (1, 0.0, 0.001),  # затухание весов оптимизатора
                'warmup_epochs': (1, 0.0, 5.0),  # количество эпох разогрева (можно дробные значения)
                'warmup_momentum': (1, 0.0, 0.95),  # начальный моментум при разогреве
                'warmup_bias_lr': (1, 0.0, 0.2),  # начальная скорость обучения для смещения при разогреве
                'box': (1, 0.02, 0.2),  # коэффициент потерь для рамок
                'cls': (1, 0.2, 4.0),  # коэффициент потерь для классификации
                'cls_pw': (1, 0.5, 2.0),  # положительный вес BCELoss для классификации
                'obj': (1, 0.2, 4.0),  # коэффициент потерь для объектов (масштабируется по пикселям)
                'obj_pw': (1, 0.5, 2.0),  # положительный вес BCELoss для объектов
                'iou_t': (0, 0.1, 0.7),  # порог IoU для обучения
                'anchor_t': (1, 2.0, 8.0),  # порог для множества якорных рамок
                'anchors': (2, 2.0, 10.0),  # количество якорных рамок на выходной сетке (0 - игнорировать)
                'fl_gamma': (0, 0.0, 2.0),
                # гамма - параметр для фокальной потери (эффективный Det по умолчанию gamma = 1.5)
                'hsv_h': (1, 0.0, 0.1),  # усиление HSV - оттенка изображения (доля)
                'hsv_s': (1, 0.0, 0.9),  # усиление HSV - насыщенности изображения (доля)
                'hsv_v': (1, 0.0, 0.9),  # усиление HSV - значения изображения (доля)
                'degrees': (1, 0.0, 45.0),  # вращение изображения (+/- градусы)
                'translate': (1, 0.0, 0.9),  # смещение изображения (+/- доля)
                'scale': (1, 0.0, 0.9),  # масштабирование изображения (+/- коэффициент)
                'shear': (1, 0.0, 10.0),  # сдвиг изображения (+/- градусы)
                'perspective': (0, 0.0, 0.001),  # перспектива изображения (+/- доля), диапазон 0 - 0.001
                'flipud': (1, 0.0, 1.0),  # переворот изображения вверх ногами (вероятность)
                'fliplr': (0, 0.0, 1.0),  # переворот изображения слева направо (вероятность)
                'mosaic': (1, 0.0, 1.0),  # смешивание изображений (вероятность)
                'mixup': (1, 0.0, 1.0),  # смешивание изображений (вероятность)
                'copy_paste': (1, 0.0, 1.0)}  # копирование - вставка сегментов (вероятность)

            # Открытие файла с гиперпараметрами и загрузка их в словарь
            with open(opt.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # загрузка словаря гиперпараметров
                # Если в гиперпараметрах нет ключа 'anchors', то устанавливаем его значение равным 3
                if 'anchors' not in hyp:  # якорные рамки закомментированы в hyp.yaml
                    hyp['anchors'] = 3
            # Если параметр noautoanchor установлен, то удаляем якорные рамки из гиперпараметров и метаданных
            if opt.noautoanchor:
                del hyp['anchors'], meta['anchors']
            # Установка параметров для эволюции гиперпараметров
            opt.noval, opt.nosave, save_dir = True, True, Path(
                opt.save_dir)  # только валидация/сохранение последней эпохи
            # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # индексы для эволюции
            # Создание путей для файлов с эволюционированными гиперпараметрами
            evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
            # Если указан бакет, то скачиваем файл evolve.csv из него
            if opt.bucket:
                os.system(
                    f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # загрузка evolve.csv, если он существует

            # Цикл для эволюции гиперпараметров
            for _ in range(opt.evolve):  # поколения для эволюции
                # Если файл evolve.csv существует, то выбираем лучшие гиперпараметры и мутируем их
                if evolve_csv.exists():  # если evolve.csv существует: выбираем лучшие гиперпараметры и мутируем
                    # Выбор родителя(ей)
                    parent = 'single'  # метод выбора родителя: 'single' или 'weighted'
                    # Загрузка данных из evolve.csv
                    x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                    # Определение количества предыдущих результатов для рассмотрения
                    n = min(5, len(x))  # количество предыдущих результатов для рассмотрения
                    # Выбор топ - n мутаций
                    x = x[np.argsort(-fitness(x))][:n]  # топ n мутаций
                    # Вычисление весов
                    w = fitness(x) - fitness(x).min() + 1E-6  # веса (сумма > 0)
                    # Выбор родителя одним из методов
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # случайный выбор
                        # Взвешенный выбор
                        x = x[random.choices(range(n), weights=w)[0]]
                    elif parent == 'weighted':
                        # Взвешенное объединение
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()

                    # Мутация
                    mp, s = 0.8, 0.2  # вероятность мутации, сигма
                    npr = np.random
                    # Установка семени для генератора случайных чисел
                    npr.seed(int(time.time()))
                    # Вычисление коэффициентов мутации
                    g = np.array([meta[k][0] for k in hyp.keys()])  # коэффициенты 0 - 1
                    # Количество гиперпараметров для эволюции
                    ng = len(meta)
                    # Инициализация вектора мутаций
                    v = np.ones(ng)
                    # Мутация до тех пор, пока не произойдет изменение (предотвращение дубликатов)
                    while all(v == 1):
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                    # Применение мутации к гиперпараметрам
                    for i, k in enumerate(hyp.keys()):
                        hyp[k] = float(x[i + 7] * v[i])  # мутация

            # Ограничение значений гиперпараметров в пределах допустимых значений
            for k, v in meta.items():
                # Установка значения гиперпараметра не ниже нижнего предела
                hyp[k] = max(hyp[k], v[1])  # нижний предел
                # Установка значения гиперпараметра не выше верхнего предела
                hyp[k] = min(hyp[k], v[2])  # верхний предел
                # Округление значения гиперпараметра до 5 знаков после запятой
                hyp[k] = round(hyp[k], 5)  # количество значащих цифр

            # Обучение с мутировавшими гиперпараметрами
            results = train(hyp.copy(), opt, device, callbacks)
            # Создание нового объекта обратных вызовов
            callbacks = Callbacks()
            # Запись результатов мутации в файл
            print_mutation(KEYS, results, hyp.copy(), save_dir, opt.bucket)

            # Построение графика эволюции гиперпараметров
            plot_evolve(evolve_csv)
            # Вывод информации о завершении эволюции гиперпараметров
            LOGGER.info(f'Эволюция гиперпараметров завершена после {opt.evolve} поколений\n'
                        f"Результаты сохранены в {colorstr('bold', save_dir)}\n"
                        f'Пример использования: $ python train.py --hyp {evolve_yaml}')

            def run(**kwargs):
                # Использование: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
                # Парсинг аргументов командной строки с флагом для внутреннего вызова
                opt = parse_opt(True)
                # Установка параметров из переданных аргументов
                for k, v in kwargs.items():
                    setattr(opt, k, v)
                # Запуск основного процесса обучения
                main(opt)
                return opt

            if __name__ == "__main__":
                # Парсинг аргументов командной строки
                opt = parse_opt()
                # Запуск основного процесса обучения
                main(opt)