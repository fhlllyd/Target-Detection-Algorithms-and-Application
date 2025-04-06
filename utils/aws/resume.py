# Resume all interrupted trainings in yolov5/ dir including DDP trainings
# Usage: $ python utils/aws/resume.py

import os
import sys
from pathlib import Path

import torch
import yaml

# Получаем абсолютный путь к текущему файлу
FILE = Path(__file__).resolve()
# Определяем корневую директорию YOLOv5 (два уровня выше текущего файла)
ROOT = FILE.parents[2]
# Если путь к корневой директории не находится в списке путей系统
if str(ROOT) not in sys.path:
    # Добавляем путь к корневой директории в список путей系统
    sys.path.append(str(ROOT))

# Инициализация номера порта равным 0 (для дальнейшей настройки_master_port в распределенном обучении)
port = 0
# Получаем абсолютный путь к текущей рабочей директории
path = Path('').resolve()
# Рекурсивно ищем файл с именем last.pt в текущей рабочей директории и ее поддиректориях
for last in path.rglob('*/**/last.pt'):
    # Загружаем чекпоинт модели
    ckpt = torch.load(last)
    # Если оптимизатор в чекпоинте модели равен None, то пропускаем текущую итерацию
    if ckpt['optimizer'] is None:
        continue

    # Загружаем файл конфигурации opt.yaml, соответствующий чекпоинту модели
    with open(last.parent.parent / 'opt.yaml', errors='ignore') as f:
        opt = yaml.safe_load(f)

    # Получаем список устройств из конфигурации (разделенные запятой)
    d = opt['device'].split(',')
    # Определяем количество устройств
    nd = len(d)
    # Определяем, является ли обучение распределенным между несколькими GPU (DDL - Distributed Data Parallel)
    ddp = nd > 1 or (nd == 0 and torch.cuda.device_count() > 1)

    if ddp:  # Если это распределенное обучение на нескольких GPU
        # Увеличиваем номер порта на 1
        port += 1
        # Создаем команду для запуска распределенного обучения с использованием torch.distributed.run
        cmd = f'python -m torch.distributed.run --nproc_per_node {nd} --master_port {port} train.py --resume {last}'
    else:  # Если это обучение на одной GPU
        # Создаем команду для запуска обучения на одной GPU, указывая чекпоинт для возобновления
        cmd = f'python train.py --resume {last}'

    # Перенаправляем вывод команды в /dev/null и запускаем команду в фоновом потоке (демоне)
    cmd += ' > /dev/null 2>&1 &'
    # Выводим сформированную команду
    print(cmd)
    # Выполняем сформированную команду
    os.system(cmd)