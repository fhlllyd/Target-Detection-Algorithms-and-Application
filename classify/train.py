# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 classifier model on a classification dataset

Usage - Single-GPU training:
    $ python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3

Datasets:           --data mnist, fashion-mnist, cifar10, cifar100, imagenette, imagewoof, imagenet, or 'path/to/data'
YOLOv5-cls models:  --model yolov5n-cls.pt, yolov5s-cls.pt, yolov5m-cls.pt, yolov5l-cls.pt, yolov5x-cls.pt
Torchvision models: --model resnet50, efficientnet_b0, etc. See https://pytorch.org/vision/stable/models.html
"""

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from tqdm import tqdm

# Получаем абсолютный путь текущего файла
FILE = Path(__file__).resolve()
# Определяем корневую директорию YOLOv5
ROOT = FILE.parents[1]
# Добавляем корневую директорию в путь поиска модулей, если ее там еще нет
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# Получаем относительный путь к корневой директории
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Импортируем функцию валидации из модуля classify
from classify import val as validate
# Импортируем функцию для загрузки модели из модуля models.experimental
from models.experimental import attempt_load
# Импортируем классы моделей из модуля models.yolo
from models.yolo import ClassificationModel, DetectionModel
# Импортируем функцию для создания загрузчика данных из модуля utils.dataloaders
from utils.dataloaders import create_classification_dataloader
# Импортируем различные утилиты из модуля utils.general
from utils.general import (DATASETS_DIR, LOGGER, TQDM_BAR_FORMAT, WorkingDirectory, check_git_info, check_git_status,
                           check_requirements, colorstr, download, increment_path, init_seeds, print_args, yaml_save)
# Импортируем класс логгера из модуля utils.loggers
from utils.loggers import GenericLogger
# Импортируем функцию для отображения изображений из модуля utils.plots
from utils.plots import imshow_cls
# Импортируем различные утилиты для работы с PyTorch из модуля utils.torch_utils
from utils.torch_utils import (ModelEMA, model_info, reshape_classifier_output, select_device, smart_DDP,
                               smart_optimizer, smartCrossEntropyLoss, torch_distributed_zero_first)

# Определяем локальный ранг процесса
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
# Определяем глобальный ранг процесса
RANK = int(os.getenv('RANK', -1))
# Определяем общее количество процессов
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
# Получаем информацию о Git-репозитории
GIT_INFO = check_git_info()

# Функция для обучения модели
def train(opt, device):
    # Инициализируем генератор случайных чисел
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # Распаковываем параметры обучения
    save_dir, data, bs, epochs, nw, imgsz, pretrained = \
        opt.save_dir, Path(opt.data), opt.batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers), \
        opt.imgsz, str(opt.pretrained).lower() == 'true'
    # Проверяем, используем ли мы CUDA
    cuda = device.type != 'cpu'

    # Создаем директорию для сохранения весов модели
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    # Определяем пути к последним и лучшим весам модели
    last, best = wdir / 'last.pt', wdir / 'best.pt'

    # Сохраняем настройки запуска в файл
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Создаем логгер, если текущий процесс главный
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # Загружаем набор данных
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        # Определяем путь к набору данных
        data_dir = data if data.is_dir() else (DATASETS_DIR / data)
        if not data_dir.is_dir():
            LOGGER.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
            t = time.time()
            if str(data) == 'imagenet':
                subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
            else:
                url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{data}.zip'
                download(url, dir=data_dir.parent)
            s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
            LOGGER.info(s)

    # Создаем загрузчики данных для обучения и валидации
    # Определяем количество классов в наборе данных
    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])
    trainloader = create_classification_dataloader(path=data_dir / 'train',
                                                   imgsz=imgsz,
                                                   batch_size=bs // WORLD_SIZE,
                                                   augment=True,
                                                   cache=opt.cache,
                                                   rank=LOCAL_RANK,
                                                   workers=nw)

    # Определяем путь к набору данных для тестирования
    test_dir = data_dir / 'test' if (data_dir / 'test').exists() else data_dir / 'val'
    if RANK in {-1, 0}:
        testloader = create_classification_dataloader(path=test_dir,
                                                      imgsz=imgsz,
                                                      batch_size=bs // WORLD_SIZE * 2,
                                                      augment=False,
                                                      cache=opt.cache,
                                                      rank=-1,
                                                      workers=nw)

    # Загружаем модель
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        if Path(opt.model).is_file() or opt.model.endswith('.pt'):
            model = attempt_load(opt.model, device='cpu', fuse=False)
        elif opt.model in torchvision.models.__dict__:
            model = torchvision.models.__dict__[opt.model](weights='IMAGENET1K_V1' if pretrained else None)
        else:
            m = hub.list('ultralytics/yolov5')
            raise ModuleNotFoundError(f'--model {opt.model} not found. Available models are: \n' + '\n'.join(m))
        if isinstance(model, DetectionModel):
            LOGGER.warning("WARNING ⚠️ pass YOLOv5 classifier model with '-cls' suffix, i.e. '--model yolov5s-cls.pt'")
            model = ClassificationModel(model=model, nc=nc, cutoff=opt.cutoff or 10)
        reshape_classifier_output(model, nc)
    for m in model.modules():
        if not pretrained and hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        if isinstance(m, torch.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout
    for p in model.parameters():
        p.requires_grad = True
    model = model.to(device)

    # Выводим информацию о модели
    if RANK in {-1, 0}:
        model.names = trainloader.dataset.classes
        model.transforms = testloader.dataset.torch_transforms
        model_info(model)
        if opt.verbose:
            LOGGER.info(model)
        images, labels = next(iter(trainloader))
        file = imshow_cls(images[:25], labels[:25], names=model.names, f=save_dir / 'train_images.jpg')
        logger.log_images(file, name='Train Examples')
        logger.log_graph(model, imgsz)

    # Создаем оптимизатор
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay)

    # Создаем планировщик скорости обучения
    lrf = 0.01
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Создаем экземпляр EMA (Exponential Moving Average)
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Включаем режим распределенной тренировки, если необходимо
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Начинаем тренировку
    t0 = time.time()
    criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)
    best_fitness = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    val = test_dir.stem
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} test\n'
                f'Using {nw * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting {opt.model} training on {data} dataset with {nc} classes for {epochs} epochs...\n\n'
                f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>12}{'top1_acc':>12}{'top5_acc':>12}")
    for epoch in range(epochs):
        tloss, vloss, fitness = 0.0, 0.0, 0.0
        model.train()
        if RANK != -1:
            trainloader.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format=TQDM_BAR_FORMAT)
        for i, (images, labels) in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device)

            # Прямой проход
            with amp.autocast(enabled=cuda):
                loss = criterion(model(images), labels)

            # Обратный проход
            scaler.scale(loss).backward()

            # Оптимизация
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

                # Прогресс
                if RANK in {-1, 0}:
                    # Получение информации о занятой памяти видеокарты
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                    # Обновление описания прогресс-бара
                    pbar.set_description(f'{epoch + 1:>10}/{epochs:>10}{mem:>10}{loss.item():>12.3g}')

                # Планировщик
                scheduler.step()

                # Валидация
                if RANK in {-1, 0}:
                    # Вычисление потерь и метрик на валидационной выборке
                    results = validate(model=ema.ema, dataloader=testloader, criterion=criterion, pbar=pbar)
                    vloss, top1, top5 = results[:3]
                    fitness = top1  # Определение фитнеса как точности top1

                    # Логирование
                    logger.log_metrics(
                        {
                            "train/loss": tloss,
                            f"{val}/loss": vloss,
                            f"{val}/top1": top1,
                            f"{val}/top5": top5,
                            "lr/0": optimizer.param_groups[0]["lr"],
                        },
                        epoch=epoch,
                    )

                    # Обновление наилучшего фитнеса
                    if fitness > best_fitness:
                        best_fitness = fitness

                    # Сохранение модели
                    final_epoch = epoch + 1 == epochs
                    if (not opt.nosave) or final_epoch:
                        ckpt = {
                            "epoch": epoch,
                            "best_fitness": best_fitness,
                            "model": deepcopy(ema.ema).half(),
                            "ema": None,
                            "updates": ema.updates,
                            "optimizer": optimizer.state_dict(),
                            "opt": vars(opt),
                            "git": GIT_INFO,
                        }

                        # Сохранение последней и лучшей моделей
                        torch.save(ckpt, last)
                        if best_fitness == fitness:
                            torch.save(ckpt, best)
                        del ckpt

                # Обучение завершено
                if RANK in {-1, 0}:
                    LOGGER.info(f'\nОбучение завершено ({time.time() - t0:.3f}s)\n'
                                f"Результаты сохранены в {colorstr('bold', save_dir)}\n"
                                f"Предсказание:         python classify/predict.py --weights {best} --source im.jpg\n"
                                f"Валидация:        python classify/val.py --weights {best} --data {data}\n"
                                f"Экспорт:          python export.py --weights {best} --include onnx\n"
                                f"PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{best}')\n"
                                f"Визуализация:       https://netron.app\n")

                return model

                def parse_opt(known=False):
                    # Создание парсера аргументов командной строки
                    parser = argparse.ArgumentParser()
                    # Определение параметра модели
                    parser.add_argument('--model', type=str, default='yolov5s-cls.pt', help='Путь к начальным весам')
                    # Определение параметра набора данных
                    parser.add_argument('--data', type=str, default='imagenette160', help='Путь к файлу dataset.yaml')
                    # Определение параметра количества эпох
                    parser.add_argument('--epochs', type=int, default=10, help='Общее количество эпох обучения')
                    # Определение параметра размера пакета
                    parser.add_argument('--batch-size', type=int, default=64, help='Общий размер пакета для всех GPU')
                    # Определение параметра размера изображения
                    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224,
                                        help='Размер изображения для обучения и тестирования (пиксели)')
                    # Определение параметра сохранения модели
                    parser.add_argument('--nosave', action='store_true',
                                        help='Сохранять только финальную контрольную точку')
                    # Определение параметра кэширования изображений
                    parser.add_argument('--cache', type=str, nargs='?', const='ram',
                                        help='Кэшировать изображения в "ram" (по умолчанию) или "disk"')
                    # Определение параметра устройства
                    parser.add_argument('--device', default='', help='Устройство CUDA, например 0 или 0,1,2,3 или cpu')
                    # Определение параметра количества рабочих потоков
                    parser.add_argument('--workers', type=int, default=8,
                                        help='Максимальное количество рабочих потоков загрузчика данных (на каждый RANK в режиме DDP)')
                    # Определение параметра проекта
                    parser.add_argument('--project', default=ROOT / 'runs/train-cls', help='Сохранять в project/name')
                    # Определение параметра имени проекта
                    parser.add_argument('--name', default='exp', help='Сохранять в project/name')
                    # Определение параметра существования проекта
                    parser.add_argument('--exist-ok', action='store_true',
                                        help='Существующий проект/имя ок, не инкрементировать')
                    # Определение параметра предобученной модели
                    parser.add_argument('--pretrained', nargs='?', const=True, default=True,
                                        help='Начинать с, например --pretrained False')
                    # Определение параметра оптимизатора
                    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam',
                                        help='Оптимизатор')
                    # Определение параметра начального скорости обучения
                    parser.add_argument('--lr0', type=float, default=0.001, help='Начальная скорость обучения')
                    # Определение параметра затухания весов
                    parser.add_argument('--decay', type=float, default=5e-5, help='Затухание весов')
                    # Определение параметра сглаживания меток
                    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Эпсилон сглаживания меток')
                    # Определение параметра индекса разрыва слоя модели
                    parser.add_argument('--cutoff', type=int, default=None,
                                        help='Индекс разрыва слоя модели для головы Classify()')
                    # Определение параметра вероятности дропаута
                    parser.add_argument('--dropout', type=float, default=None, help='Вероятность дропаута')
                    # Определение параметра подробного режима
                    parser.add_argument('--verbose', action='store_true', help='Подробный режим')
                    # Определение параметра семени генератора случайных чисел
                    parser.add_argument('--seed', type=int, default=0, help='Глобальное семя обучения')
                    # Определение параметра локального ранга
                    parser.add_argument('--local_rank', type=int, default=-1,
                                        help='Автоматический аргумент DDP Multi-GPU, не изменять')

                    # Парсинг аргументов
                    return parser.parse_known_args()[0] if known else parser.parse_args()

                def main(opt):
                    # Проверки
                    if RANK in {-1, 0}:
                        print_args(vars(opt))
                        check_git_status()
                        check_requirements()

                    # Режим DDP
                    device = select_device(opt.device, batch_size=opt.batch_size)
                    if LOC_RANK != -1:
                        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size должен быть кратным количеству устройств CUDA'
                        assert not opt.image_weights, '--image-weights аргумент несовместим с обучением DDP'
                        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

                    # Обучение
                    train(opt, device)

                if __name__ == "__main__":
                    opt = parse_opt()
                    main(opt)