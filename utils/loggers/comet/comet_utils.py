import logging
import os
from urllib.parse import urlparse

try:
    # Попытка импортировать модуль comet_ml
    import comet_ml
except (ModuleNotFoundError, ImportError):
    # Если модуль не найден или произошла ошибка импорта, установить comet_ml в None
    comet_ml = None

import yaml

# Инициализация логгера с именем текущего модуля
logger = logging.getLogger(__name__)

# Префикс для идентификации ресурсов Comet
COMET_PREFIX = "comet://"
# Имя модели по умолчанию, полученное из переменной окружения или установленное в "yolov5"
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")
# Имя файла чекпоинта по умолчанию, полученное из переменной окружения или установленное в "last.pt"
COMET_DEFAULT_CHECKPOINT_FILENAME = os.getenv("COMET_DEFAULT_CHECKPOINT_FILENAME", "last.pt")


def download_model_checkpoint(opt, experiment):
    # Создание директории для сохранения модели
    model_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(model_dir, exist_ok=True)

    # Получение имени модели
    model_name = COMET_MODEL_NAME
    # Получение списка активов (ассетов) модели из эксперимента Comet
    model_asset_list = experiment.get_model_asset_list(model_name)

    # Если список активов модели пуст, логируем ошибку и выходим из функции
    if len(model_asset_list) == 0:
        logger.error(f"COMET ERROR: No checkpoints found for model name : {model_name}")
        return

    # Сортировка списка активов модели по шагу (step) в обратном порядке
    model_asset_list = sorted(
        model_asset_list,
        key=lambda x: x["step"],
        reverse=True,
    )
    # Создание словаря, где ключ - имя файла, значение - идентификатор актива
    logged_checkpoint_map = {asset["fileName"]: asset["assetId"] for asset in model_asset_list}

    # Разбор URL весов модели
    resource_url = urlparse(opt.weights)
    # Получение имени файла чекпоинта из запроса URL
    checkpoint_filename = resource_url.query

    if checkpoint_filename:
        # Получение идентификатора актива по имени файла чекпоинта
        asset_id = logged_checkpoint_map.get(checkpoint_filename)
    else:
        # Если имя файла чекпоинта не указано, используем имя по умолчанию
        asset_id = logged_checkpoint_map.get(COMET_DEFAULT_CHECKPOINT_FILENAME)
        checkpoint_filename = COMET_DEFAULT_CHECKPOINT_FILENAME

    # Если идентификатор актива не найден, логируем ошибку и выходим из функции
    if asset_id is None:
        logger.error(f"COMET ERROR: Checkpoint {checkpoint_filename} not found in the given Experiment")
        return

    try:
        # Логируем информацию о начале загрузки чекпоинта
        logger.info(f"COMET INFO: Downloading checkpoint {checkpoint_filename}")
        asset_filename = checkpoint_filename

        # Получение бинарного содержимого актива (чекпоинта)
        model_binary = experiment.get_asset(asset_id, return_type="binary", stream=False)
        # Путь для сохранения загруженного чекпоинта
        model_download_path = f"{model_dir}/{asset_filename}"
        # Сохранение бинарного содержимого в файл
        with open(model_download_path, "wb") as f:
            f.write(model_binary)

        # Обновление пути к весам модели на путь к загруженному чекпоинту
        opt.weights = model_download_path

    except Exception as e:
        # Логируем предупреждение и исключение, если не удалось загрузить чекпоинт
        logger.warning("COMET WARNING: Unable to download checkpoint from Comet")
        logger.exception(e)


def set_opt_parameters(opt, experiment):
    """
    Обновляет пространство имен opt параметрами из существующего эксперимента Comet при возобновлении запуска.

    Аргументы:
        opt (argparse.Namespace): Пространство имен командной строки опций.
        experiment (comet_ml.APIExperiment): Объект эксперимента Comet API.
    """
    # Получение списка активов эксперимента
    asset_list = experiment.get_asset_list()
    # Сохранение строки resume из opt
    resume_string = opt.resume

    for asset in asset_list:
        if asset["fileName"] == "opt.yaml":
            # Получение идентификатора актива для файла opt.yaml
            asset_id = asset["assetId"]
            # Получение бинарного содержимого файла opt.yaml
            asset_binary = experiment.get_asset(asset_id, return_type="binary", stream=False)
            # Загрузка содержимого файла opt.yaml в словарь
            opt_dict = yaml.safe_load(asset_binary)
            for key, value in opt_dict.items():
                # Установка атрибутов в opt на основе значений из словаря
                setattr(opt, key, value)
            # Возвращение исходной строки resume в opt
            opt.resume = resume_string

    # Создание директории для сохранения гиперпараметров
    save_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(save_dir, exist_ok=True)

    # Путь к файлу гиперпараметров в формате YAML
    hyp_yaml_path = f"{save_dir}/hyp.yaml"
    with open(hyp_yaml_path, "w") as f:
        # Сохранение гиперпараметров в файл YAML
        yaml.dump(opt.hyp, f)
    # Обновление пути к гиперпараметрам в opt
    opt.hyp = hyp_yaml_path


def check_comet_weights(opt):
    """
    Загружает веса модели из Comet и обновляет путь к весам на сохраненное местоположение.

    Аргументы:
        opt (argparse.Namespace): Аргументы командной строки, переданные в скрипт обучения YOLOv5.

    Возвращает:
        None/bool: Возвращает True, если веса успешно загружены, в противном случае возвращает None.
    """
    if comet_ml is None:
        return

    if isinstance(opt.weights, str):
        if opt.weights.startswith(COMET_PREFIX):
            # Создание объекта API Comet
            api = comet_ml.API()
            # Разбор URL весов модели
            resource = urlparse(opt.weights)
            # Получение пути к эксперименту из URL
            experiment_path = f"{resource.netloc}{resource.path}"
            # Получение объекта эксперимента Comet
            experiment = api.get(experiment_path)
            # Загрузка чекпоинта модели
            download_model_checkpoint(opt, experiment)
            return True

    return None


def check_comet_resume(opt):
    """
    Восстанавливает параметры запуска до исходного состояния на основе чекпоинта модели
    и зарегистрированных параметров эксперимента.

    Аргументы:
        opt (argparse.Namespace): Аргументы командной строки, переданные в скрипт обучения YOLOv5.

    Возвращает:
        None/bool: Возвращает True, если запуск успешно восстановлен, в противном случае возвращает None.
    """
    if comet_ml is None:
        return

    if isinstance(opt.resume, str):
        if opt.resume.startswith(COMET_PREFIX):
            # Создание объекта API Comet
            api = comet_ml.API()
            # Разбор URL для возобновления запуска
            resource = urlparse(opt.resume)
            # Получение пути к эксперименту из URL
            experiment_path = f"{resource.netloc}{resource.path}"
            # Получение объекта эксперимента Comet
            experiment = api.get(experiment_path)
            # Установка параметров opt на основе эксперимента
            set_opt_parameters(opt, experiment)
            # Загрузка чекпоинта модели
            download_model_checkpoint(opt, experiment)

            return True

    return None