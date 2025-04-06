import sys
from pathlib import Path

import wandb

FILE = Path(__file__).resolve()
# Получение абсолютного пути к текущему файлу и сохранение в переменную FILE
ROOT = FILE.parents[3]  # Корневая директория YOLOv5
# Получение родительской директории на 3 уровня выше текущего файла (корневой директории YOLOv5) и сохранение в ROOT
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Добавление ROOT в системный путь

from train import parse_opt, train
# Импорт функций parse_opt и train из модуля train
from utils.callbacks import Callbacks
# Импорт класса Callbacks из модуля utils.callbacks
from utils.general import increment_path
# Импорт функции increment_path из модуля utils.general
from utils.torch_utils import select_device
# Импорт функции select_device из модуля utils.torch_utils


def sweep():
    wandb.init()
    # Инициализация сессии Wandb

    # Получение словаря гиперпараметров из конфигурации Wandb и создание его копии
    hyp_dict = vars(wandb.config).get("_items").copy()

    # Получение необходимых параметров опций
    opt = parse_opt(known=True)
    opt.batch_size = hyp_dict.get("batch_size")
    # Определение директории для сохранения
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve))
    opt.epochs = hyp_dict.get("epochs")
    opt.nosave = True
    opt.data = hyp_dict.get("data")
    opt.weights = str(opt.weights)
    opt.cfg = str(opt.cfg)
    opt.data = str(opt.data)
    opt.hyp = str(opt.hyp)
    opt.project = str(opt.project)
    # Выбор устройства для обучения
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Запуск обучения
    train(hyp_dict, opt, device, callbacks=Callbacks())


if __name__ == "__main__":
    sweep()
    # Если скрипт запускается напрямую, то вызывается функция sweep