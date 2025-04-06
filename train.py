# Импорт необходимых модулей Python и сторонних библиотек
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

# Установка пути к проекту
FILE = Path(__file__).resolve()           # Получение пути к текущему скрипту (FILE) и его корневой директории (ROOT)
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:             # Добавление ROOT в системный путь, если его там нет
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Преобразование пути ROOT в относительный

# Импорт пользовательских модулей для поддержки обучения
import val as validate
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

# Инициализация переменных для распределенного обучения
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))     # Локальный ранг процесса (используется в DDP)
RANK = int(os.getenv('RANK', -1))      # Глобальный ранг процесса
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))   # Общее количество процессов
GIT_INFO = check_git_info()     # Информация о состоянии Git репозитория

# Определение функции обучения
def train(hyp, opt, device, callbacks):  # Основная функция обучения YOLOv5
                                         # hyp: Конфигурация гиперпараметров (путь к .yaml или словарь)
                                         # opt: Параметры командной строки
                                         # device: Устройство (CPU/GPU)
                                         # callbacks: Экземпляр回调-функций

    # Извлечение параметров обучения из opt
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
    Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
    opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')    # Вызов события перед началом обучения

    # Настройка директорий для сохранения весов
    w = save_dir / 'weights'
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    # Загрузка гиперпараметров
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    LOGGER.info(colorstr('гиперпараметры: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()

    # Сохранение настроек эксперимента
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Инициализация логировщиков
    data_dict = None
    if RANK in {-1, 0}:   # Для главного процесса
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # Создание экземпляра логировщика

        # Регистрация действий логировщиков
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Обработка удаленного датасета
        data_dict = loggers.remote_dataset
        if resume:
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Конфигурация среды обучения
    plots = not evolve and not opt.noplots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)

    # Извлечение путей к датасетам
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')

    # Инициализация модели
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu')
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'Перенесено {len(csd)}/{len(model.state_dict())} элементов из {weights}')
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    amp = check_amp(model)

    # Заморозка слоев модели
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            LOGGER.info(f'заморозка {k}')
            v.requires_grad = False

    # Настройка размеров изображений и батча
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Конфигурация оптимизатора
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Конфигурация планировщика обучения
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Инициализация экспоненциального сглаживания (EMA)
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Возобновление обучения
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # Конфигурация Data Parallel (DP)
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # Синхронная нормализация батча
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Использование SyncBatchNorm()')

    # Создание загрузчика тренировочных данных
    train_loader, dataset = create_dataloader(train_path,
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
                                              shuffle=True)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # Максимальный класс меток
    assert mlc < nc, f'Класс метки {mlc} превышает nc={nc} в {data}. Допустимые классы: 0-{nc - 1}'

# В основном процессе выполняется создание загрузчика валидационных данных и некоторые предварительные операции обучения.
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
                                       prefix=colorstr('val: '))[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # Запуск AutoAnchor
            model.half().float()

        callbacks.run('on_pretrain_routine_end', labels, names)

    # Включение режима распределенной параллельной обработки данных (DDP).
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Установка атрибутов модели.
    nl = de_parallel(model).model[-1].nl
    hyp['box'] *= 3 / nl
    hyp['cls'] *= nc / 80 * 3 / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # Начало обучения, инициализация переменных, связанных с обучением, и запись информации о начале обучения.
    t0 = time.time()
    nb = len(train_loader)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)
    callbacks.run('on_train_start')
    LOGGER.info(f'Размеры изображений {imgsz} для обучения, {imgsz} для валидации\n'
                f'Использование {train_loader.num_workers * WORLD_SIZE} рабочих процессов загрузчика данных\n'
                f"Запись результатов в {colorstr('bold', save_dir)}\n"
                f'Запуск обучения на {epochs} эпох...')
    # Выполнение цикла обучения
    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()

        # Если включены веса изображений (--image-weights), то обновить веса изображений на основе весов классов.
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)

        # Обновление границ мозаики (по желанию)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # Границы по высоте и ширине

        # Инициализация средней потери, установка семплера загрузчика данных, создание полосы прогресса и сброс градиентов оптимизатора.
        mloss = torch.zeros(3, device=device)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Эпоха', 'Память GPU', 'Потеря рамок', 'Потеря объекта', 'Потеря класса', 'Экземпляры', 'Размер'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()
        # Итерация по каждому пакету данных, вызов функций обратного вызова, обновление номера пакета и нормализация изображений.
        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # Фаза разогрева: корректировка скорости обучения и момента, чтобы модель постепенно адаптировалась к изменению скорости обучения и избежала взрыва или исчезновения градиента.
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Случайное изменение размера изображений для многошкальной тренировки, чтобы повысить адаптивность модели к разным масштабам изображений и обобщающую способность модели.
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Использование автоматического смешанного представления точности (AMP) для прямого распространения, вычисление потери и корректировка потери в зависимости от режима распределенного обучения.
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))
                if RANK != -1:
                    loss *= WORLD_SIZE
                if opt.quad:
                    loss *= 4.

            # Масштабирование потери и выполнение обратного распространения
            scaler.scale(loss).backward()

            # Выполнение шага оптимизации, включая отмену масштабирования градиента, обрезку градиента, обновление оптимизатора и модели EMA, чтобы обеспечить стабильность процесса оптимизации и повысить производительность и обобщающую способность модели.
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # В основном процессе записать журнальные данные, включая среднюю потерю, использование памяти видеокарты и т.д.
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return

        # Обновление планировщика скорости обучения, чтобы адаптироваться к разным этапам процесса обучения.
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        # Оценка производительности модели на валидационном наборе, вычисление ключевых показателей (например, mAP) и принятие решения о раннем прекращении обучения в зависимости от функции обратного вызова.
        if RANK in {-1, 0}:
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Вычисление mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            # Обновление записи лучших показателей на основе текущих показателей производительности и запись ключевых журнальных данных в процессе обучения с помощью функции обратного вызова.
            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Сохранение модели, сохранение состояния модели, состояния оптимизатора и других данных в файл для последующего восстановления обучения или оценки модели.
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
                    'date': datetime.now().isoformat()}

                # Сохранение состояния модели, состояния оптимизатора и других данных в файл для последующего восстановления обучения или оценки модели.
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # Предотвращение переобучения с помощью механизма раннего прекращения и обеспечение синхронизации остановки всех процессов в распределенном обучении.
        if RANK != -1:  # Если выполняется распределенное обучение (DDP)
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break

# Завершение обучения, запись журнальных данных, проверка лучшей модели и очистка памяти видеокарты.
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results

# Определение аргументов командной строки для настройки обучения.
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # --weights: Путь к начальным весам модели
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    # --cfg: Путь к конфигурационному файлу модели
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # --data: Путь к конфигурационному файлу датасета
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    # --hyp: Путь к гиперпараметрам обучения
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    # --epochs: Количество эпох обучения
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    # --batch-size: Размер пакета для всех GPU
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    # --imgsz: Размер изображений для обучения
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # --rect: Использование прямоугольных изображений
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # --resume: Продолжение обучения из чекпоинта
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # --nosave: Сохранение только финального чекпоинта
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # --noval: Пропуск валидации до последней эпохи
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # --noautoanchor: Отключение автоматического подбора якорных боксов
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    # --noplots: Отключение сохранения графиков
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    # --evolve: Эволюция гиперпараметров
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # --bucket: Название хранилища Google Cloud
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # --cache: Кэширование изображений в памяти или на диске
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    # --image-weights: Взвешенный выбор изображений для обучения
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # --device: Устройство для обучения (GPU или CPU)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # --multi-scale: Множественный масштаб изображений
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # --single-cls: Обработка данных как одиночный класс
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # --optimizer: Выбор оптимизатора
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    # --sync-bn: Синхронная нормализация батча
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # --workers: Количество рабочих процессов загрузчика данных
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    # --project: Путь к проекту для сохранения результатов
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    # --name: Имя эксперимента
    parser.add_argument('--name', default='exp', help='save to project/name')
    # --exist-ok: Продолжение эксперимента при совпадении имени
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # --quad: Четырехкратный даталоадер
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # --cos-lr: Косинусный график изменения скорости обучения
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    # --label-smoothing: Параметр сглаживания меток
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # --patience: Количество эпох без улучшения для раннего останова
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    # --freeze: Заморозка указанных слоев
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    # --save-period: Интервал сохранения чекпоинтов
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    # --seed: Глобальный сид для воспроизводимости
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    # --local_rank: Параметр для распределенного обучения
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    # --entity: Организация для логирования в Weights & Biases
    parser.add_argument('--entity', default=None, help='Entity')
    # --upload_dataset: Загрузка датасета на сервер
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    # --bbox_interval: Интервал логирования bounding-box
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    # --artifact_alias: Версия артефакта датасета
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')
    return parser.parse_known_args()[0] if known else parser.parse_args()

# Основная функция
def main(opt, callbacks=Callbacks()):
    # Проверка процесса
    if RANK in {-1, 0}:
        print_args(vars(opt))   # Вывод параметров командной строки
        check_git_status()      # Проверка статуса Git
        check_requirements()    # Проверка зависимостей

    # Проверка необходимости возобновления обучения, при условии, что это не возобновление Comet и не эволюция гиперпараметров
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())  # Если opt.resume - строка, то это путь к файлу чекпоинта; в противном случае - путь к последнему запуску обучения
        opt_yaml = last.parent.parent / 'opt.yaml'   # Построение пути к файлу конфигурации параметров для возобновления обучения
        opt_data = opt.data    # Сохранение исходного пути к датасету
        # Если файл конфигурации параметров существует
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:  # Открытие файла конфигурации параметров с игнорированием ошибок кодировки
                d = yaml.safe_load(f)   # Безопасная загрузка данных YAML из файла, получение предыдущей конфигурации параметров обучения
        # Если файл конфигурации параметров не существует
        else:
            d = torch.load(last, map_location='cpu')['opt']   # Загрузка данных из файла чекпоинта и отображение их на CPU, затем извлечение предыдущей конфигурации параметров обучения
        opt = argparse.Namespace(**d)     # Применение предыдущей конфигурации параметров обучения к текущему объекту opt
        opt.cfg, opt.weights, opt.resume = '', str(last), True   # Очистка пути к файлу конфигурации модели, установка пути к весам в путь к файлу чекпоинта, явное указание на возобновление обучения
        # Если исходный путь к датасету - URL
        if is_url(opt_data):
            opt.data = check_file(opt_data)   # Проверка и загрузка файла датасета по URL, обновление opt.data на локальный путь к файлу

    # Если условия для возобновления обучения не выполняются
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)   # Проверка валидности
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'  # Убедиться, что указан хотя бы файл конфигурации модели или файл весов

        # Если включена эволюция гиперпараметров
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # Если путь к проекту - по умолчанию runs/train, то изменить на runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # Разрешить существование пути к проекту и не возобновлять обучение

        # Если имя - 'cfg', то использовать имя файла конфигурации модели в качестве имени
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)) # Генерация пути к директории сохранения, обеспечение уникальности пути

    # Конфигурация режима DDP, проверка совместимости параметров, настройка устройства, инициализация группы процессов для правильной настройки распределенного обучения на нескольких GPU.
    # Выбор устройства для обучения (CPU или GPU)
    device = select_device(opt.device, batch_size=opt.batch_size)
    ## Если текущий процесс - часть распределенного обучения на нескольких GPU
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'     # Определение сообщения об ошибке для указания, что некоторые параметры несовместимы с распределенным обучением на нескольких GPU DDP
        assert not opt.image_weights, f'--image-weights {msg}'    # Убедиться, что не используется взвешенный выбор изображений для обучения
        assert not opt.evolve, f'--evolve {msg}'     # Убедиться, что не выполняется эволюция гиперпараметров
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'    # Убедиться, что не используется автоматический размер пакета
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'    # Убедиться, что размер пакета - кратный количеству процессов (WORLD_SIZE)
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'    # Убедиться, что количество доступных CUDA-устройств больше локального ранга текущего процесса
        torch.cuda.set_device(LOCAL_RANK)        # Установка текущего CUDA-устройства для процесса
        device = torch.device('cuda', LOCAL_RANK)      # Обновление объекта устройства на текущее CUDA-устройство для процесса
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")     # Инициализация группы распределенных процессов, выбор nccl или gloo в качестве бэкенда в зависимости от доступности


# В зависимости от того, включено ли эволюционное обучение (--evolve), выбирается выполнение обычного обучения или эволюционного обучения.

    # Обычное обучение
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Эволюционное обучение. Определяется диапазон поиска гиперпараметров и подготавливается к эволюции гиперпараметров.
    else:
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # Начальная скорость обучения
            'lrf': (1, 0.01, 1.0),  # Конечная скорость обучения
            'momentum': (0.3, 0.6, 0.98),  # Импульс SGD
            'weight_decay': (1, 0.0, 0.001),  # Сокращение весов оптимизатора
            'warmup_epochs': (1, 0.0, 5.0),  # Количество эпох разогрева
            'warmup_momentum': (1, 0.0, 0.95),  # Начальный импульс разогрева
            'warmup_bias_lr': (1, 0.0, 0.2),  # Начальная скорость обучения смещения при разогреве
            'box': (1, 0.02, 0.2),  # Коэффициент потерь для рамок
            'cls': (1, 0.2, 4.0),  # Коэффициент потерь для классов
            'cls_pw': (1, 0.5, 2.0),   # Положительный вес BCELoss для классов
            'obj': (1, 0.2, 4.0),  # Коэффициент потерь для объектов (масштабируется по пикселям)
            'obj_pw': (1, 0.5, 2.0),   # Положительный вес BCELoss для объектов
            'iou_t': (0, 0.1, 0.7),  # Порог IoU при обучении
            'anchor_t': (1, 2.0, 8.0),  # Порог кратности якорных рамок
            'anchors': (2, 2.0, 10.0),   # Количество якорных рамок для каждой выходной сетки (0 означает игнорировать)
            'fl_gamma': (0, 0.0, 2.0),   # Гамма для фокусной потери
            'hsv_h': (1, 0.0, 0.1),  # Усиление оттенка (HSV-Hue) изображения
            'hsv_s': (1, 0.0, 0.9),  # Усиление насыщенности (HSV-Saturation) изображения
            'hsv_v': (1, 0.0, 0.9),  # Усиление яркости (HSV-Value) изображения
            'degrees': (1, 0.0, 45.0),  # Поворот изображения (+/- градусов)
            'translate': (1, 0.0, 0.9),   # Перемещение изображения (+/- доля)
            'scale': (1, 0.0, 0.9),  # Масштабирование изображения (+/- коэффициент)
            'shear': (1, 0.0, 10.0),  # Сдвиг изображения (+/- градусов)
            'perspective': (0, 0.0, 0.001),  # Перспектива изображения (+/- доля), диапазон 0 - 0.001
            'flipud': (1, 0.0, 1.0),   # Вертикальное отражение изображения (вероятность)
            'fliplr': (0, 0.0, 1.0),   # Горизонтальное отражение изображения (вероятность)
            'mosaic': (1, 0.0, 1.0),  # Использование мозаики в аугментации изображений (вероятность)
            'mixup': (1, 0.0, 1.0),  # Использование MixUp в аугментации изображений (вероятность)
            'copy_paste': (1, 0.0, 1.0)}  # Использование копирования и вставки сегментации в аугментации изображений (вероятность)

        # Загрузка файла конфигурации гиперпараметров
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)   # Безопасная загрузка конфигурации гиперпараметров из файла
            if 'anchors' not in hyp:  # Если в конфигурации гиперпараметров нет поля 'anchors', добавить его и установить значение 3
                hyp['anchors'] = 3

        # Если пользователь указал опцию отключения автоматических якорных рамок (--noautoanchor)
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']    # Удалить поля 'anchors'
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)     # Валидация только на последней эпохе (noval) - True
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'  # Определение пути к файлу конфигурации эволюционированных гиперпараметров (в формате .yaml)

        # Если пользователь указал имя бакета gsutil (opt.bucket не пусто)
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}') # Копирование файла evolve.csv из бакета в локальную директорию сохранения

        # Выполнение цикла эволюционного обучения, оптимизация гиперпараметров с помощью случайных мутаций и отбора.
        for _ in range(opt.evolve):
            # Если файл с записями результатов эволюции (evolve.csv) существует
            if evolve_csv.exists():
                parent = 'single'      # Установка стратегии выбора родителя на 'single' (выбор одного родителя)
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)    # Загрузка данных из файла evolve.csv, обеспечение, чтобы данные были как минимум двумерным массивом, использование запятой в качестве разделителя, пропуск первой строки (заголовок)
                n = min(5, len(x))     # Взятие первых n строк данных, где n - минимальное из 5 и количества строк данных
                x = x[np.argsort(-fitness(x))][:n]    # Сортировка данных по убыванию приспособленности (fitness) и взятие первых n строк
                w = fitness(x) - fitness(x).min() + 1E-6   # Вычисление разности приспособленностей и добавление малой константы 1E-6, чтобы избежать деления на ноль
                if parent == 'single' or len(x) == 1:       # Если выбрана стратегия выбора одного родителя или данных только одна строка, случайно выбрать один из n данных в качестве родителя с учетом весов приспособленности
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':     # Если выбрана стратегия взвешенного среднего, выполнить взвешенное среднее по n данным, чтобы получить новый родительский набор данных
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()

                mp, s = 0.8, 0.2   # Установка вероятности мутации mp равной 0.8 и масштабного коэффициента s равным 0.2
                npr = np.random   # Получение генератора случайных чисел numpy
                npr.seed(int(time.time()))   # Использование текущего времени в качестве семени для генератора случайных чисел
                g = np.array([meta[k][0] for k in hyp.keys()])    # Получение массива значений по умолчанию g из диапазона поиска гиперпараметров (meta) на основе ключей конфигурации гиперпараметров
                ng = len(meta)    # Получение длины диапазона поиска гиперпараметров
                v = np.ones(ng)  # Создание массива v длиной ng, заполненного единицами
                # Пока все элементы в v равны 1, продолжать цикл по генерации мутаций, сгенерировать мутации v с учетом вероятности mp и случайных чисел, и обрезать значения в диапазоне от 0.3 до 3.0
                while all(v == 1):
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                # Перебор ключей конфигурации гиперпараметров, обновить конфигурацию гиперпараметров на основе родительского набора данных и мутаций
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])

            # Перебор диапазона поиска гиперпараметров (meta)
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])   # Гарантировать, что значение гиперпараметра не меньше нижней границы диапазона поиска
                hyp[k] = min(hyp[k], v[2])   # Гарантировать, что значение гиперпараметра не больше верхней границы диапазона поиска
                hyp[k] = round(hyp[k], 5)    # Округлить значение гиперпараметра до 5 знаков после запятой
            # Обучение с обновленной конфигурацией гиперпараметров и возврат результатов обучения
            results = train(hyp.copy(), opt, device, callbacks)
            # Переинициализация объекта функций обратного вызова
            callbacks = Callbacks()
            # Определение ключей показателей, которые будут записаны и напечатаны
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')
            # Вывод информации о мутации, включая показатели, результаты, конфигурацию гиперпараметров, директорию сохранения и имя бакета
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Построение кривой эволюции на основе файла с записями результатов эволюции (evolve.csv)
        plot_evolve(evolve_csv)
        # Запись информации о завершении эволюции гиперпараметров, включая количество поколений, директорию сохранения результатов и пример использования
        LOGGER.info(f'Эволюция гиперпараметров завершена за {opt.evolve} поколений\n'
                    f"Результаты сохранены в {colorstr('bold', save_dir)}\n"
                    f'Пример использования: $ python train.py --hyp {evolve_yaml}')

# Определение функции run, которая используется для запуска скрипта обучения с помощью ключевых аргументов.
def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

# Точка входа в основной скрипт. Парсинг аргументов командной строки и запуск обучения.
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

