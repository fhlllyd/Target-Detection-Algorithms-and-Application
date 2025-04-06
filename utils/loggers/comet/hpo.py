import argparse
import json
import logging
import os
import sys
from pathlib import Path

import comet_ml

logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # Корневая директория YOLOv5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  #Добавление корневой директории в системный путь

from train import train
from utils.callbacks import Callbacks
from utils.general import increment_path
from utils.torch_utils import select_device

# Конфигурация проекта
config = comet_ml.config.get_config()
COMET_PROJECT_NAME = config.get_string(os.getenv("COMET_PROJECT_NAME"), "comet.project_name", default="yolov5")


def get_args(known=False):
    parser = argparse.ArgumentParser()
    # Путь к начальным весам модели
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    # Путь к конфигурационному файлу модели
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # Путь к конфигурационному файлу датасета
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    # Путь к файлу гиперпараметров
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    # Общее количество эпох обучения
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    # Размер батча для всех GPU, -1 для автоматического определения
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    # Размер изображений для обучения и валидации (пиксели)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # Флаг для прямоугольного обучения
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # Флаг для возобновления последнего обучения
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # Флаг для сохранения только последнего чекпоинта
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # Флаг для валидации только на последней эпохе
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # Флаг для отключения AutoAnchor
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    # Флаг для сохранения без файлов с графиками
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    # Количество поколений для эволюции гиперпараметров
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # Название gsutil bucket
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # Кэширование изображений в "ram" или "disk"
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    # Флаг для использования взвешенного выбора изображений для обучения
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # Устройство (cuda или cpu)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # Флаг для изменения размера изображений на +/- 50%
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # Флаг для обучения много классных данных как едино классных
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # Оптимизатор (SGD, Adam, AdamW)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    # Флаг для использования SyncBatchNorm (только в DDP режиме)
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # Максимальное количество воркеров дата로адера (за RANK в DDP режиме)
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    # Директория для сохранения результатов (project/name)
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    # Имя для сохранения результатов (project/name)
    parser.add_argument('--name', default='exp', help='save to project/name')
    # Флаг для того, чтобы проект/имя было допустимым, не нужно инкрементировать
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # Флаг для quatro дата로адера
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # Флаг для косинусногоcheduler LR
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    # Эпсилон для сглаживания меток
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # Период нетерпения для ранней остановки (эпохи без улучшения)
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    # Слои для заморозки (например, backbone=10, first3=0 1 2)
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    # Период сохранения чекпоинта (запрещено, если < 1)
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    # Глобальный сид для обучения
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    # Аргумент для автоматического DDP Multi-GPU, не нужно изменять
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Аргументы Weights & Biases
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    # Флаг для загрузки датасета в W&B, опция "val"
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    # Интервал для логирования изображений с ограничивающими рамками в W&B
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    # псевдоним артефакта для использования в W&B
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    # Аргументы Comet
    parser.add_argument("--comet_optimizer_config", type=str, help="Comet: Path to a Comet Optimizer Config File.")
    parser.add_argument("--comet_optimizer_id", type=str, help="Comet: ID of the Comet Optimizer sweep.")
    parser.add_argument("--comet_optimizer_objective", type=str, help="Comet: Set to 'minimize' or 'maximize'.")
    parser.add_argument("--comet_optimizer_metric", type=str, help="Comet: Metric to Optimize.")
    parser.add_argument("--comet_optimizer_workers",
                        type=int,
                        default=1,
                        help="Comet: Number of Parallel Workers to use with the Comet Optimizer.")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(parameters, opt):
    # Создание словаря гиперпараметров, исключая "epochs" и "batch_size"
    hyp_dict = {k: v for k, v in parameters.items() if k not in ["epochs", "batch_size"]}

    # Определение директории для сохранения результатов
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve))
    # Установка размера батча из параметров
    opt.batch_size = parameters.get("batch_size")
    # Установка количества эпох из параметров
    opt.epochs = parameters.get("epochs")

    # Выбор устройства (cuda или cpu)
    device = select_device(opt.device, batch_size=opt.batch_size)
    # Запуск обучения с заданными гиперпараметрами, опциями, устройством и обратными вызовами
    train(hyp_dict, opt, device, callbacks=Callbacks())


if __name__ == "__main__":
    opt = get_args(known=True)

    opt.weights = str(opt.weights)
    opt.cfg = str(opt.cfg)
    opt.data = str(opt.data)
    opt.project = str(opt.project)

    optimizer_id = os.getenv("COMET_OPTIMIZER_ID")
    if optimizer_id is None:
        # Чтение конфигурационного файла оптимизатора Comet
        with open(opt.comet_optimizer_config) as f:
            optimizer_config = json.load(f)
        optimizer = comet_ml.Optimizer(optimizer_config)
    else:
        optimizer = comet_ml.Optimizer(optimizer_id)

    opt.comet_optimizer_id = optimizer.id
    status = optimizer.status()

    opt.comet_optimizer_objective = status["spec"]["objective"]
    opt.comet_optimizer_metric = status["spec"]["metric"]

    logger.info("COMET INFO: Starting Hyperparameter Sweep")
    for parameter in optimizer.get_parameters():
        run(parameter["parameters"], opt)