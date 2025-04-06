# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Общие утилиты
"""

import contextlib
import glob
import inspect
import logging
import logging.config
import math
import os
import platform
import random
import re
import signal
import sys
import time
import urllib
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from typing import Optional
from zipfile import ZipFile, is_zipfile

import cv2
import IPython
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

from utils import TryExcept, emojis
from utils.downloads import gsutil_getsize
from utils.metrics import box_iou, fitness

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Корневой каталог YOLOv5
RANK = int(os.getenv('RANK', -1))

# Настройки
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # Количество потоков multiprocessing для YOLOv5
DATASETS_DIR = Path(os.getenv('YOLOv5_DATASETS_DIR', ROOT.parent / 'datasets'))  # Глобальный каталог наборов данных
AUTOINSTALL = str(os.getenv('YOLOv5_AUTOINSTALL', True)).lower() == 'true'  # Глобальный режим автоматической установки
VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # Глобальный режим verbose
TQDM_BAR_FORMAT = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}'  # Формат строки tqdm
FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf

torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # Формат короткого g, точность=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # Предотвратить многопоточность OpenCV (несовместима с PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # Максимальное количество потоков NumExpr
os.environ['OMP_NUM_THREADS'] = '1' if platform.system() == 'darwin' else str(NUM_THREADS)  # Потоки OpenMP (PyTorch и SciPy)


def is_ascii(s=''):
    # Проверяет, состоит ли строка только из символов ASCII
    s = str(s)  # Преобразование списка, кортежа, None и т.д. в строку
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    # Проверяет, состоит ли строка из китайских символов
    return bool(re.search('[\u4e00-\u9fff]', str(s)))


def is_colab():
    # Проверяет, является ли среда экземпляром Google Colab
    return 'google.colab' in sys.modules


def is_notebook():
    # Проверяет, является ли среда Jupyter Notebook
    ipython_type = str(type(IPython.get_ipython()))
    return 'colab' in ipython_type or 'zmqshell' in ipython_type


def is_kaggle():
    # Проверяет, является ли среда Kaggle Notebook
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'


def is_docker() -> bool:
    """Проверяет, запущен ли процесс внутри контейнера Docker."""
    if Path("/.dockerenv").exists():
        return True
    try:  # Проверка наличия Docker в группах контроля
        with open("/proc/self/cgroup") as file:
            return any("docker" in line for line in file)
    except OSError:
        return False


def is_writeable(dir, test=False):
    # Возвращает True, если каталог имеет права записи, проверяет возможность открытия файла с правами записи, если test=True
    if not test:
        return os.access(dir, os.W_OK)  # Возможны проблемы на Windows
    file = Path(dir) / 'tmp.txt'
    try:
        with open(file, 'w'):  # Открывает файл с правами записи
            pass
        file.unlink()  # Удаляет файл
        return True
    except OSError:
        return False


LOGGING_NAME = "yolov5"


def set_logging(name=LOGGING_NAME, verbose=True):
    # Настройка журнала с заданным именем
    rank = int(os.getenv('RANK', -1))  # ранг в мире для многопроцессорных тренировок
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(message)s"}},
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level,}},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False,}}})


set_logging(LOGGING_NAME)  # запуск перед определением LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # определение глобально
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # безопасный журнал с эмодзи


def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # Возвращает путь к каталогу пользовательской конфигурации. Предпочтительно использовать переменную среды, если она существует. Создает каталог, если необходимо.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # использовать переменную среды
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # каталоги для 3 ОС
        path = Path.home() / cfg.get(platform.system(), '')  # каталог OS
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # исправление для GCP и AWS lambda, только /tmp доступен для записи
    path.mkdir(exist_ok=True)  # создать каталог, если необходимо
    return path


CONFIG_DIR = user_config_dir()  # каталог настроек Ultralytics


class Profile(contextlib.ContextDecorator):
    # Класс профилирования YOLOv5. Использование: @Profile() декоратор или 'with Profile():' контекстный менеджер
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # накопление dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


class Timeout(contextlib.ContextDecorator):
    # Класс таймаута YOLOv5. Использование: @Timeout(seconds) декоратор или 'with Timeout(seconds):' контекстный менеджер
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        if platform.system() != 'Windows':  # не поддерживается на Windows
            signal.signal(signal.SIGALRM, self._timeout_handler)  # Установка обработчика SIGALRM
            signal.alarm(self.seconds)  # запуск обратного отсчета для SIGALRM

    def __exit__(self, exc_type, exc_val, exc_tb):
        if platform.system() != 'Windows':
            signal.alarm(0)  # Отмена SIGALRM, если она запланирована
            if self.suppress and exc_type is TimeoutError:  # Подавление ошибок таймаута
                return True


class WorkingDirectory(contextlib.ContextDecorator):
    # Использование: @WorkingDirectory(dir) декоратор или 'with WorkingDirectory(dir):' контекстный менеджер
    def __init__(self, new_dir):
        self.dir = new_dir  # новый каталог
        self.cwd = Path.cwd().resolve()  # текущий каталог

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)


def methods(instance):
    # Получение методов класса/экземпляра
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith("__")]


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Вывод аргументов функции (необязательный словарь args)
    x = inspect.currentframe().f_back  # предыдущий фрейм
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # автоматическое получение аргументов
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))


def init_seeds(seed=0, deterministic=False):
    # Инициализация генераторов случайных чисел
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # безопасно для многопроцессорных GPU
    if deterministic and check_version(torch.__version__, '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)


def intersect_dicts(da, db, exclude=()):
    # Пересечение словарей с соответствующими ключами и формами, исключая 'exclude' ключи, используя значения da
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def get_default_args(func):
    # Получение значений по умолчанию для func()
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_latest_run(search_dir='.'):
    # Возвращает путь к последнему 'last.pt' в /runs (например, для --resume)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def file_age(path=__file__):
    # Возвращает количество дней с последнего изменения файла
    dt = (datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime))  # delta
    return dt.days  # + dt.seconds / 86400  # дробные дни


def file_date(path=__file__):
    # Возвращает человеческий формат даты изменения файла, например, '2021-3-26'
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def file_size(path):
    # Возвращает размер файла/каталога (МБ)
    mb = 1 << 20  # байты в МиБ (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0


def check_online():
    # Проверка подключения к интернету
    import socket

    def run_once():
        # Проверка один раз
        try:
            socket.create_connection(("1.1.1.1", 443), 5)  # проверка доступности хоста
            return True
        except OSError:
            return False

    return run_once() or run_once()  # проверка дважды для увеличения устойчивости к временным проблемам подключения


def git_describe(path=ROOT):  # path должен быть каталогом
    # Возвращает человекочитаемое описание git, например, v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    try:
        assert (Path(path) / '.git').is_dir()
        return check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    except Exception:
        return ''


@TryExcept()
@WorkingDirectory(ROOT)
def check_git_status(repo='ultralytics/yolov5', branch='master'):
    # Состояние YOLOv5, рекомендуется использовать 'git pull', если код устарел
    url = f'https://github.com/{repo}'
    msg = f', для обновления см. {url}'
    s = colorstr('github: ')  # строка
    assert Path('.git').is_dir(), s + 'пропуск проверки (не является git-репозиторием)' + msg
    assert check_online(), s + 'пропуск проверки (оффлайн)' + msg

    splits = re.split(pattern=r'\s', string=check_output('git remote -v', shell=True).decode())
    matches = [repo in s for s in splits]
    if any(matches):
        remote = splits[matches.index(True) - 1]
    else:
        remote = 'ultralytics'
        check_output(f'git remote add {remote} {url}', shell=True)
    check_output(f'git fetch {remote}', shell=True, timeout=5)  # git fetch
    local_branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # текущая ветка
    n = int(check_output(f'git rev-list {local_branch}..{remote}/{branch} --count', shell=True))  # количество коммитов позади
    if n > 0:
        pull = 'git pull' if remote == 'origin' else f'git pull {remote} {branch}'
        s += f"⚠️ YOLOv5 устарело на {n} коммит{'ов' if n > 1 else ''}. Используйте `{pull}` или `git clone {url}` для обновления."
    else:
        s += f'обновлено до последней версии с {url} ✅'
    LOGGER.info(s)


@WorkingDirectory(ROOT)
def check_git_info(path='.'):
    # Информация о YOLOv5 git, возвращает {remote, branch, commit}
    check_requirements('gitpython')
    import git
    try:
        repo = git.Repo(path)
        remote = repo.remotes.origin.url.replace('.git', '')  # например, 'https://github.com/ultralytics/yolov5'
        commit = repo.head.commit.hexsha  # например, '3134699c73af83aac2a481435550b968d5792c0d'
        try:
            branch = repo.active_branch.name  # например, 'main'
        except TypeError:  # не на какой-либо ветке
            branch = None  # например, состояние 'detached HEAD'
        return {'remote': remote, 'branch': branch, 'commit': commit}
    except git.exc.InvalidGitRepositoryError:  # путь не является git-каталогом
        return {'remote': None, 'branch': None, 'commit': None}


def check_python(minimum='3.7.0'):
    # Проверка текущей версии Python на соответствие минимальной версии
    check_version(platform.python_version(), minimum, name='Python ', hard=True)


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Проверка версии на соответствие минимальной
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING ⚠️ {name}{minimum} требуется для YOLOv5, но текущая версия {name}{current}'  # строка
    if hard:
        assert result, emojis(s)  # утверждение минимальных требований
    if verbose and not result:
        LOGGER.warning(s)
    return result


@TryExcept()
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True, cmds=''):
    # Проверка установленных зависимостей на соответствие требованиям YOLOv5 (передать *.txt файл или список пакетов или строку с одним пакетом)
    prefix = colorstr('red', 'bold', 'требования:')
    check_python()  # проверка версии Python
    if isinstance(requirements, Path):  # requirements.txt файл
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} не найден, проверка завершена."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    s = ''
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except (pkg.VersionConflict, pkg.DistributionNotFound):  # исключение, если требования не выполнены
            s += f'"{r}" '
            n += 1

    if s and install and AUTOINSTALL:  # проверка переменной среды
        LOGGER.info(f"{prefix} Требования YOLOv5 {s}не найдены, попытка автоматического обновления...")
        try:
            # assert check_online(), "Пропуск автоматического обновления (оффлайн)"
            LOGGER.info(check_output(f'pip install {s} {cmds}', shell=True).decode())
            source = file if 'file' in locals() else requirements
            s = f"{prefix} {n} пакет{'а' if n % 10 == 1 and n % 100 != 11 else 'ов' if n % 10 in [2, 3, 4] and n % 100 not in [12, 13, 14] else ''} обновлен{'о' if n == 1 else 'ы'} в соответствии с {source}\n" \
                f"{prefix} ⚠️ {colorstr('bold', 'Перезапустите среду выполнения или повторно выполните команду для вступления изменений в силу')}\n"
            LOGGER.info(s)
        except Exception as e:
            LOGGER.warning(f'{prefix} ❌ {e}')


def check_img_size(imgsz, s=32, floor=0):
    # Проверка, что размер изображения кратен шагу s в каждой размерности
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} должен быть кратен максимальному шагу {s}, обновление до {new_size}')
    return new_size


def check_imshow(warn=False):
    # Проверка, поддерживает ли среда отображение изображений
    try:
        assert not is_notebook()
        assert not is_docker()
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f'WARNING ⚠️ Среда не поддерживает cv2.imshow() или PIL Image.show()\n{e}')
        return False


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Проверка расширения файла
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # расширение файла
            if len(s):
                assert s in suffix, f"{msg}{f} допустимое расширение {suffix}"


def check_yaml(file, suffix=('.yaml', '.yml')):
    # Поиск/скачивание файла YAML (если необходимо) и возврат пути, проверка расширения
    return check_file(file, suffix)


def check_file(file, suffix=''):
    # Поиск/скачивание файла (если необходимо) и возврат пути
    check_suffix(file, suffix)  # проверка расширения
    file = str(file)  # преобразование в строку
    if os.path.isfile(file) or not file:  # существует
        return file
    elif file.startswith(('http:/', 'https:/')):  # скачать
        url = file  # предупреждение: Pathlib превращает :// в :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' в '/', разделение https://url.com/file.txt?auth
        if os.path.isfile(file):
            LOGGER.info(f"{file} найден локально")  # файл уже существует
        else:
            LOGGER.info(f'Скачивание {url} в {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'Ошибка загрузки файла: {url}'  # проверка
        return file
    elif file.startswith('clearml://'):  # ClearML Dataset ID
        assert 'clearml' in sys.modules, "ClearML не установлен, поэтому нельзя использовать набор данных ClearML. Попробуйте выполнить 'pip install clearml'."
        return file
    else:  # поиск
        files = []
        for d in 'data', 'models', 'utils':  # каталоги поиска
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # поиск файла
        assert len(files), f'Файл не найден: {file}'  # утверждение, что файл найден
        assert len(files) == 1, f"Найдено несколько файлов, соответствующих '{file}', уточните путь: {files}"  # утверждение уникальности
        return files[0]  # возврат файла


def check_font(font=FONT, progress=False):
    # Скачивание шрифта в CONFIG_DIR, если необходимо
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = f'https://ultralytics.com/assets/{font.name}'
        LOGGER.info(f'Скачивание {url} в {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=progress)


def check_dataset(data, autodownload=True):
    # Скачивание, проверка и/или распаковка набора данных, если он не найден локально

    # Скачивание (опционально)
    extract_dir = ''
    if isinstance(data, (str, Path)) and (is_zipfile(data) or is_tarfile(data)):
        download(data, dir=f'{DATASETS_DIR}/{Path(data).stem}', unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Чтение yaml (опционально)
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  # словарь

    # Проверки
    for k in 'train', 'val', 'names':
        assert k in data, emojis(f"В data.yaml отсутствует поле '{k}:'")
    if isinstance(data['names'], (list, tuple)):  # старый формат массива
        data['names'] = dict(enumerate(data['names']))  # преобразование в словарь
    assert all(isinstance(k, int) for k in data['names'].keys()), 'Ключи в data.yaml должны быть целыми числами, например, 2: car'
    data['nc'] = len(data['names'])

    # Разрешение путей
    path = Path(extract_dir or data.get('path') or '')  # необязательный 'path' по умолчанию '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data['path'] = path  # каталог загрузки сценариев
    for k in 'train', 'val', 'test':
        if data.get(k):  # предварительно добавление пути
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Парсинг yaml
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in val]  # val путь
        if not all(x.exists() for x in val):
            LOGGER.info('\nНабор данных не найден ⚠️, отсутствуют пути %s' % [str(x) for x in val if not x.exists()])
            if not s or not autodownload:
                raise Exception('Набор данных не найден ❌')
            t = time.time()
            if s.startswith('http') and s.endswith('.zip'):  # URL
                f = Path(s).name  # имя файла
                LOGGER.info(f'Скачивание {s} в {f}...')
                torch.hub.download_url_to_file(s, f)
                Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)  # создание корневого каталога
                unzip_file(f, path=DATASETS_DIR)  # распаковка
                Path(f).unlink()  # удаление zip
                r = None  # успешность
            elif s.startswith('bash '):  # bash скрипт
                LOGGER.info(f'Выполнение {s} ...')
                r = os.system(s)
            else:  # python скрипт
                r = exec(s, {'yaml': data})  # возвращает None
            dt = f'({round(time.time() - t, 1)}с)'
            s = f"успешно ✅ {dt}, сохранено в {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f"неудачно {dt} ❌"
            LOGGER.info(f"Загрузка набора данных {s}")
    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf', progress=True)  # загрузка шрифтов
    return data  # словарь


def check_amp(model):
    # Проверка функциональности автоматической смешанной точности (AMP) PyTorch. Возвращает True при успешной работе
    from models.common import AutoShape, DetectMultiBackend

    def amp_allclose(model, im):
        # Проверка близости FP32 и AMP результатов
        m = AutoShape(model, verbose=False)  # модель
        a = m(im).xywhn[0]  # вывод FP32
        m.amp = True
        b = m(im).xywhn[0]  # вывод AMP
        return a.shape == b.shape and torch.allclose(a, b, atol=0.1)  # близость с точностью 10%

    prefix = colorstr('AMP: ')
    device = next(model.parameters()).device  # устройство модели
    if device.type in ('cpu', 'mps'):
        return False  # AMP используется только на CUDA устройствах
    f = ROOT / 'data' / 'images' / 'bus.jpg'  # изображение для проверки
    im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if check_online() else np.ones((640, 640, 3))
    try:
        assert amp_allclose(deepcopy(model), im) or amp_allclose(DetectMultiBackend('yolov5n.pt', device), im)
        LOGGER.info(f'{prefix}проверки завершены успешно ✅')
        return True
    except Exception:
        help_url = 'https://github.com/ultralytics/yolov5/issues/7908'
        LOGGER.warning(f'{prefix}проверки завершены с ошибкой ❌, отключение автоматической смешанной точности. Смотрите {help_url}')
        return False


def yaml_load(file='data.yaml'):
    # Безопасная загрузка YAML в одну строку
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def yaml_save(file='data.yaml', data={}):
    # Безопасная запись YAML в одну строку
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX')):
    # Распаковка *.zip файла в path/, исключая файлы, содержащие строки в exclude
    if path is None:
        path = Path(file).parent  # каталог по умолчанию
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # список всех именованных файлов в zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)


def url2file(url):
    # Преобразование URL в имя файла, например, https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib превращает :// в :/
    return Path(urllib.parse.unquote(url)).name.split('?')[0]  # '%2F' в '/', разделение https://url.com/file.txt?auth


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1, retry=3):
    # Многопоточная функция загрузки и распаковки файлов, используемая в data.yaml для автоматической загрузки
    def download_one(url, dir):
        # Загрузка 1 файла
        success = True
        if os.path.isfile(url):
            f = Path(url)  # имя файла
        else:  # отсутствует
            f = dir / Path(url).name
            LOGGER.info(f'Скачивание {url} в {f}...')
            for i in range(retry + 1):
                if curl:
                    s = 'sS' if threads > 1 else ''  # тихий режим
                    r = os.system(f'curl -# -{s}L "{url}" -o "{f}"')
                    success = r == 0
                else:
                    torch.hub.download_url_to_file(url, f, progress=threads == 1)  # загрузка torch
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    LOGGER.warning(f'⚠️ Ошибка загрузки, повторная попытка {i + 1}/{retry} {url}...')
                else:
                    LOGGER.warning(f'❌ Не удалось скачать {url}...')

        if unzip and success and (f.suffix == '.gz' or is_zipfile(f) or is_tarfile(f)):
            LOGGER.info(f'Распаковка {f}...')
            if is_zipfile(f):
                unzip_file(f, dir)  # распаковка
            elif is_tarfile(f):
                os.system(f'tar xf {f} --directory {f.parent}')  # распаковка
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  # распаковка
            if delete:
                f.unlink()  # удаление zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # создание каталога
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # многопоточная загрузка
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    # Возвращает ближайшее к x, кратное divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # преобразование в int
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Очищает строку, заменяя специальные символы на подчеркивание _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # Лямбда-функция для синусоидального перехода от y1 к y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Окрашивание строки https://en.wikipedia.org/wiki/ANSI_escape_code, например, colorstr('blue', 'bold', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # цветовые аргументы, строка
    colors = {
        'black': '\033[30m',  # базовые цвета
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # яркие цвета
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # прочее
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Получение весов классов (обратная частота) из тренировочных меток
    if labels[0] is None:  # метки отсутствуют
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) для COCO
    classes = labels[:, 0].astype(int)  # метки классов
    weights = np.bincount(classes, minlength=nc)  # количество вхождений для каждого класса

    # Добавление количества точек сетки в начало (для uCE тренировки)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # количество точек сетки на изображении
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # добавление точек сетки в начало

    weights[weights == 0] = 1  # замена пустых бинов на 1
    weights = 1 / weights  # количество целей для каждого класса
    weights /= weights.sum()  # нормализация
    return torch.from_numpy(weights).float()


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Получение весов изображений на основе class_weights и содержимого изображений
    # Использование: index = random.choices(range(n), weights=image_weights, k=1)  # взвешенный выбор изображения
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def xyxy2xywh(x):
    # Преобразование nx4 коробок из [x1, y1, x2, y2] в [x, y, w, h], где xy1 - верхний левый угол, xy2 - нижний правый угол
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # центр x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # центр y
    y[:, 2] = x[:, 2] - x[:, 0]  # ширина
    y[:, 3] = x[:, 3] - x[:, 1]  # высота
    return y


def xywh2xyxy(x):
    # Преобразование nx4 коробок из [x, y, w, h] в [x1, y1, x2, y2], где xy1 - верхний левый угол, xy2 - нижний правый угол
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # верхний левый x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # верхний левый y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # нижний правый x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # нижний правый y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Преобразование nx4 коробок из [x, y, w, h] normalized в [x1, y1, x2, y2], где xy1 - верхний левый угол, xy2 - нижний правый угол
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # верхний левый x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # верхний левый y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # нижний правый x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # нижний правый y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Преобразование nx4 коробок из [x1, y1, x2, y2] в [x, y, w, h] normalized, где xy1 - верхний левый угол, xy2 - нижний правый угол
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # предупреждение: изменение на месте
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # центр x
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # центр y
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # ширина
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # высота
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Преобразование нормализованных сегментов в пиксельные сегменты, форма (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # верхний левый x
    y[:, 1] = h * x[:, 1] + padh  # верхний левый y
    return y


def segment2box(segment, width=640, height=640):
    # Преобразование сегмента метки 1 в коробку 1, применяя ограничение внутри изображения, то есть (xy1, xy2, ...) в (xyxy)
    x, y = segment.T  # сегмент xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Преобразование меток сегментов в метки коробок, то есть (cls, xy1, xy2, ...) в (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # сегмент xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Увеличение сегмента (n,2)
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # сегмент xy
    return segments


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Масштабирование коробок (xyxy) из img1_shape в img0_shape
    if ratio_pad is None:  # рассчитать из img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # заполнение wh
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # заполнение x
    boxes[:, [1, 3]] -= pad[1]  # заполнение y
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    # Масштабирование сегментов (xy1,xy2,...) из img1_shape в img0_shape
    if ratio_pad is None:  # рассчитать из img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # заполнение wh
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # заполнение x
    segments[:, 1] -= pad[1]  # заполнение y
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # ширина
        segments[:, 1] /= img0_shape[0]  # высота
    return segments


def clip_boxes(boxes, shape):
    # Обрезка коробок (xyxy) до размеров изображения (высота, ширина)
    if isinstance(boxes, torch.Tensor):  # быстрая обработка по отдельности
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (быстрая групповая обработка)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def clip_segments(segments, shape):
    # Обрезка сегментов (xy1,xy2,...) до размеров изображения (высота, ширина)
    if isinstance(segments, torch.Tensor):  # быстрая обработка по отдельности
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (быстрая групповая обработка)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # количество масок
):
    """Подавление немаксимумов (NMS) на результатах вывода для отклонения перекрывающихся обнаружений

    Возвращает:
         список обнаружений, на каждом изображении (n,6) тензор [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # модель YOLOv5 в режиме валидации, выход = (вывод_inference, выход_loss)
        prediction = prediction[0]  # выбрать только вывод inference

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS не полностью поддерживается, преобразовать тензоры в CPU перед NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # размер пакета
    nc = prediction.shape[2] - nm - 5  # количество классов
    xc = prediction[..., 4] > conf_thres  # кандидаты

    # Проверки
    assert 0 <= conf_thres <= 1, f'Недопустимая пороговая Confidence {conf_thres}, допустимые значения между 0.0 и 1.0'
    assert 0 <= iou_thres <= 1, f'Недопустимая IoU {iou_thres}, допустимые значения между 0.0 и 1.0'

    # Настройки
    # min_wh = 2  # минимальная ширина и высота коробки (в пикселях)
    max_wh = 7680  # максимальная ширина и высота коробки (в пикселях)
    max_nms = 30000  # максимальное количество коробок для torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # секунды для завершения
    redundant = True  # требовать избыточные обнаружения
    multi_label &= nc > 1  # несколько меток на коробку (добавляет 0,5 мс на изображение)
    merge = False  # использовать merge-NMS

    t = time.time()
    mi = 5 + nc  # индекс начала маски
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # индекс изображения, вывод инференса
        # Применение ограничений
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # ширина-высота
        x = x[xc[xi]]  # confidence

        # Добавление априорных меток
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # коробка
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # Если нет кандидатов, обрабатывать следующее изображение
        if not x.shape[0]:
            continue

        # Вычисление conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Коробка/Маска
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) в (x1, y1, x2, y2)
        mask = x[:, mi:]  # нулевые столбцы, если нет масок

        # Матрица обнаружений nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # только лучший класс
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Фильтрация по классу
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Применение конечного ограничения
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Проверка формы
        n = x.shape[0]  # количество коробок
        if not n:  # нет коробок
            continue
        elif n > max_nms:  # избыточное количество коробок
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # сортировка по confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # сортировка по confidence

        # Подавление немаксимумов (NMS)
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # классы
        boxes, scores = x[:, :4] + c, x[:, 4]  # коробки (смещены классом), оценки
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # ограничение количества обнаружений
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge-NMS
            # обновление коробок как boxes(i,4) = взвешенное среднее boxes
            iou = box_iou(boxes[i], boxes) > iou_thres  # матрица iou
            weights = iou * scores[None]  # веса коробок
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # объединенные коробки
            if redundant:
                i = i[iou.sum(1) > 1]  # требовать избыточность

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ Время NMS {time_limit:.3f}с истекло')
            break  # превышено время ожидания

    return output


def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Удаление оптимизатора из 'f' для финализации тренировки, необязательно сохранить как 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # замена модели на ema
    for k in 'optimizer', 'best_fitness', 'ema', 'updates':  # ключи
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # преобразование в FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # размер файла
    LOGGER.info(f"Оптимизатор удален из {f},{f' сохранен как {s},' если {s} указано {mb:.1f}MB")


def print_mutation(keys, results, hyp, save_dir, bucket, prefix=colorstr('evolve: ')):
    evolve_csv = save_dir / 'evolve.csv'
    evolve_yaml = save_dir / 'hyp_evolve.yaml'
    keys = tuple(keys) + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # Загрузка (опционально)
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (evolve_csv.stat().st_size if evolve_csv.exists() else 0):
            os.system(f'gsutil cp {url} {save_dir}')  # загрузка evolve.csv, если он больше локального

    # Запись в evolve.csv
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # заголовок
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

    # Сохранение yaml
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())  # очистка ключей
        i = np.argmax(fitness(data.values[:, :4]))  #
        generations = len(data)
        f.write('# Результаты гиперпараметров YOLOv5\n' + f'# Лучшее поколение: {i}\n' +
                f'# Последнее поколение: {generations - 1}\n' + '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) +
                '\n' + '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(data.loc[i][7:].to_dict(), f, sort_keys=False)

    # Вывод на экран
    LOGGER.info(prefix + f'{generations} поколений завершено, текущий результат:\n' + prefix +
                ', '.join(f'{x.strip():>20s}' for x in keys) + '\n' + prefix + ', '.join(f'{x:20.5g}'
                                                                                         for x in vals) + '\n\n')

    if bucket:
        os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')  # загрузка


def apply_classifier(x, model, img, im0):
    # Применение вторичного классификатора к выводам YOLO
    # Пример model = torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # для каждого изображения
        if d is not None and len(d):
            d = d.clone()

            # Преобразование и заполнение вырезанных фрагментов
            b = xyxy2xywh(d[:, :4])  # коробки
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # преобразование в квадрат
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # заполнение
            d[:, :4] = xywh2xyxy(b).long()

            # Масштабирование коробок от img_size к размеру im0
            scale_boxes(img.shape[2:], d[:, :4], im0[i].shape)

            # Классы
            pred_cls1 = d[:, 5].long()
            ims = []
            for a in d:
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR в RGB, в 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 в float32
                im /= 255  # 0 - 255 в 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # предсказание классификатора
            x[i] = x[i][pred_cls1 == pred_cls2]  # сохранение соответствующих классов обнаружений

    return x


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Увеличение пути файла или каталога, например, runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... и т.д.
    path = Path(path)  # осознание OS
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Метод 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # увеличение пути
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Метод 2 (устаревший)
        # dirs = glob.glob(f"{path}{sep}*")  # похожие пути
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # индексы
        # n = max(i) + 1 if i else 2  # увеличение номера
        # path = Path(f"{path}{sep}{n}{suffix}")  # увеличение пути

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # создание каталога

    return path


# Дружественные к китайскому языку функции OpenCV ------------------------------------------------------------------------------------
imshow_ = cv2.imshow  # копирование для предотвращения рекурсивных ошибок


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def imwrite(path, im):
    try:
        cv2.imencode(Path(path).suffix, im)[1].tofile(path)
        return True
    except Exception:
        return False


def imshow(path, im):
    imshow_(path.encode('unicode_escape').decode(), im)


cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow  # переопределение