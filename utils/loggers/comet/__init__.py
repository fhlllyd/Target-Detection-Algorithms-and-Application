import glob
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # Корневая директория YOLOv5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Добавление корневой директории в системный путь

try:
    import comet_ml

    # Конфигурация проекта
    config = comet_ml.config.get_config()
    COMET_PROJECT_NAME = config.get_string(os.getenv("COMET_PROJECT_NAME"), "comet.project_name", default="yolov5")
except (ModuleNotFoundError, ImportError):
    comet_ml = None
    COMET_PROJECT_NAME = None

import PIL
import torch
import torchvision.transforms as T
import yaml

from utils.dataloaders import img2label_paths
from utils.general import check_dataset, scale_boxes, xywh2xyxy
from utils.metrics import box_iou

COMET_PREFIX = "comet://"

COMET_MODE = os.getenv("COMET_MODE", "online")

# Настройки сохранения модели
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")

# Настройки артефактов датасета
COMET_UPLOAD_DATASET = os.getenv("COMET_UPLOAD_DATASET", "false").lower() == "true"

# Настройки оценки
COMET_LOG_CONFUSION_MATRIX = os.getenv("COMET_LOG_CONFUSION_MATRIX", "true").lower() == "true"
COMET_LOG_PREDICTIONS = os.getenv("COMET_LOG_PREDICTIONS", "true").lower() == "true"
COMET_MAX_IMAGE_UPLOADS = int(os.getenv("COMET_MAX_IMAGE_UPLOADS", 100))

# Настройки матрицы ошибок
CONF_THRES = float(os.getenv("CONF_THRES", 0.001))
IOU_THRES = float(os.getenv("IOU_THRES", 0.6))

# Настройки логирования батчей
COMET_LOG_BATCH_METRICS = os.getenv("COMET_LOG_BATCH_METRICS", "false").lower() == "true"
COMET_BATCH_LOGGING_INTERVAL = os.getenv("COMET_BATCH_LOGGING_INTERVAL", 1)
COMET_PREDICTION_LOGGING_INTERVAL = os.getenv("COMET_PREDICTION_LOGGING_INTERVAL", 1)
COMET_LOG_PER_CLASS_METRICS = os.getenv("COMET_LOG_PER_CLASS_METRICS", "false").lower() == "true"

RANK = int(os.getenv("RANK", -1))

to_pil = T.ToPILImage()


class CometLogger:
    """Логирование метрик, параметров, исходного кода, моделей и многое другое
    с использованием Comet
    """

    def __init__(self, opt, hyp, run_id=None, job_type="Training", **experiment_kwargs) -> None:
        self.job_type = job_type
        self.opt = opt
        self.hyp = hyp

        # Флаги Comet
        self.comet_mode = COMET_MODE

        self.save_model = opt.save_period > -1
        self.model_name = COMET_MODEL_NAME

        # Настройки логирования батчей
        self.log_batch_metrics = COMET_LOG_BATCH_METRICS
        self.comet_log_batch_interval = COMET_BATCH_LOGGING_INTERVAL

        # Настройки артефактов датасета
        self.upload_dataset = self.opt.upload_dataset if self.opt.upload_dataset else COMET_UPLOAD_DATASET
        self.resume = self.opt.resume

        # Параметры по умолчанию для объектов Experiment
        self.default_experiment_kwargs = {
            "log_code": False,  # Отключить логирование исходного кода
            "log_env_gpu": True,  # Включить логирование GPU-информации
            "log_env_cpu": True,  # Включить логирование CPU-информации
            "project_name": COMET_PROJECT_NAME,  # Название проекта Comet
        }
        self.default_experiment_kwargs.update(
            experiment_kwargs)  # Обновление параметров с использованием дополнительных аргументов
        self.experiment = self._get_experiment(self.comet_mode, run_id)  # Инициализация эксперимента Comet

        self.data_dict = self.check_dataset(self.opt.data)  # Загрузка и проверка датасета
        self.class_names = self.data_dict["names"]  # Список названий классов
        self.num_classes = self.data_dict["nc"]  # Количество классов

        self.logged_images_count = 0  # Счетчик залогированных изображений
        self.max_images = COMET_MAX_IMAGE_UPLOADS  # Максимальное количество изображений для загрузки

        if run_id is None:
            self.experiment.log_other("Создано из", "YOLOv5")  # Логирование информации о создании эксперимента
            if not isinstance(self.experiment, comet_ml.OfflineExperiment):
                # Извлечение информации о рабочем пространстве и эксперименте из URL
                workspace, project_name, experiment_id = self.experiment.url.split("/")[-3:]
                self.experiment.log_other(
                    "Путь к эксперименту",
                    f"{workspace}/{project_name}/{experiment_id}",
                )
            self.log_parameters(vars(opt))  # Логирование параметров командной строки
            self.log_parameters(self.opt.hyp)  # Логирование гиперпараметров
            self.log_asset_data(
                self.opt.hyp,
                name="hyperparameters.json",
                metadata={"type": "hyp-config-file"},
            )  # Сохранение гиперпараметров как артефакта
            self.log_asset(
                f"{self.opt.save_dir}/opt.yaml",
                metadata={"type": "opt-config-file"},
            )  # Сохранение конфигурационного файла как артефакта

        self.comet_log_confusion_matrix = COMET_LOG_CONFUSION_MATRIX  # Флаг для логирования матрицы ошибок

        # Определение порогов для оценки
        self.conf_thres = self.opt.conf_thres if hasattr(self.opt, "conf_thres") else CONF_THRES
        self.iou_thres = self.opt.iou_thres if hasattr(self.opt, "iou_thres") else IOU_THRES

        self.log_parameters({
            "val_iou_threshold": self.iou_thres,  # Порог IoU для валидации
            "val_conf_threshold": self.conf_thres  # Порог уверенности для валидации
        })

        self.comet_log_predictions = COMET_LOG_PREDICTIONS  # Флаг для логирования предсказаний
        if self.opt.bbox_interval == -1:
            # Автоматическое определение интервала для логирования bounding box
            self.comet_log_prediction_interval = 1 if self.opt.epochs < 10 else self.opt.epochs // 10
        else:
            self.comet_log_prediction_interval = self.opt.bbox_interval  # Использование заданного интервала

        if self.comet_log_predictions:
            self.metadata_dict = {}  # Словарь для хранения метаданных изображений
            self.logged_image_names = []  # Список залогированных изображений

        self.comet_log_per_class_metrics = COMET_LOG_PER_CLASS_METRICS  # Флаг для логирования метрик по классам

        # Логирование настроек Comet в эксперимент
        self.experiment.log_others({
            "comet_mode": COMET_MODE,
            "comet_max_image_uploads": COMET_MAX_IMAGE_UPLOADS,
            "comet_log_per_class_metrics": COMET_LOG_PER_CLASS_METRICS,
            "comet_log_batch_metrics": COMET_LOG_BATCH_METRICS,
            "comet_log_confusion_matrix": COMET_LOG_CONFUSION_MATRIX,
            "comet_model_name": COMET_MODEL_NAME,
        })

        # Проверка использования оптимизатора Comet
        if hasattr(self.opt, "comet_optimizer_id"):
            self.experiment.log_other("optimizer_id", self.opt.comet_optimizer_id)
            self.experiment.log_other("optimizer_objective", self.opt.comet_optimizer_objective)
            self.experiment.log_other("optimizer_metric", self.opt.comet_optimizer_metric)
            self.experiment.log_other("optimizer_parameters",
                                      json.dumps(self.hyp))  # Логирование гиперпараметров оптимизатора

        def _get_experiment(self, mode, experiment_id=None):
            """Создание или восстановление эксперимента Comet"""
            if mode == "offline":
                # Восстановление существующего оффлайн-эксперимента или создание нового
                if experiment_id is not None:
                    return comet_ml.ExistingOfflineExperiment(
                        previous_experiment=experiment_id,
                        **self.default_experiment_kwargs,
                    )
                return comet_ml.OfflineExperiment(**self.default_experiment_kwargs)

            else:
                try:
                    # Восстановление существующего онлайн-эксперимента или создание нового
                    if experiment_id is not None:
                        return comet_ml.ExistingExperiment(
                            previous_experiment=experiment_id,
                            **self.default_experiment_kwargs,
                        )
                    return comet_ml.Experiment(**self.default_experiment_kwargs)

                except ValueError:
                    logger.warning("COMET WARNING: "
                                   "Comet credentials have not been set. "
                                   "Comet will default to offline logging. "
                                   "Please set your credentials to enable online logging.")
                    return self._get_experiment("offline", experiment_id)
                return

    import glob
    from pathlib import Path
    import torch
    from PIL import Image

    class CometLogger:
        # ...其他属性和方法定义在上文，此处省略

        def log_metrics(self, log_dict, **kwargs):
            """
            Логирует метрики в эксперименте Comet.

            :param log_dict: Словарь с метриками для логирования.
            :param kwargs: Дополнительные аргументы для log_metrics метода Comet.
            """
            self.experiment.log_metrics(log_dict, **kwargs)

        def log_parameters(self, log_dict, **kwargs):
            """
            Логирует параметры в эксперименте Comet.

            :param log_dict: Словарь с параметрами для логирования.
            :param kwargs: Дополнительные аргументы для log_parameters метода Comet.
            """
            self.experiment.log_parameters(log_dict, **kwargs)

        def log_asset(self, asset_path, **kwargs):
            """
            Логирует артефакт (например, файл) в эксперименте Comet.

            :param asset_path: Путь к артефакту.
            :param kwargs: Дополнительные аргументы для log_asset метода Comet.
            """
            self.experiment.log_asset(asset_path, **kwargs)

        def log_asset_data(self, asset, **kwargs):
            """
            Логирует данные артефакта в эксперименте Comet.

            :param asset: Данные артефакта.
            :param kwargs: Дополнительные аргументы для log_asset_data метода Comet.
            """
            self.experiment.log_asset_data(asset, **kwargs)

        def log_image(self, img, **kwargs):
            """
            Логирует изображение в эксперименте Comet.

            :param img: Изображение для логирования.
            :param kwargs: Дополнительные аргументы для log_image метода Comet.
            """
            self.experiment.log_image(img, **kwargs)

        def log_model(self, path, opt, epoch, fitness_score, best_model=False):
            """
            Логирует модель в эксперименте Comet.

            :param path: Путь к каталогу с моделями.
            :param opt: Объект с опциями (возможно, настройками обучения).
            :param epoch: Номер текущей эпохи обучения.
            :param fitness_score: Список значений "фитнес" - оценки качества модели на каждой эпохе.
            :param best_model: Флаг, указывающий, является ли модель лучшей.
            """
            if not self.save_model:
                return

            model_metadata = {
                "fitness_score": fitness_score[-1],
                "epochs_trained": epoch + 1,
                "save_period": opt.save_period,
                "total_epochs": opt.epochs,
            }

            model_files = glob.glob(f"{path}/*.pt")
            for model_path in model_files:
                name = Path(model_path).name

                self.experiment.log_model(
                    self.model_name,
                    file_or_folder=model_path,
                    file_name=name,
                    metadata=model_metadata,
                    overwrite=True,
                )

        def check_dataset(self, data_file):
            """
            Проверяет и загружает датасет. Если путь к датасету начинается с префикса Comet,
            пытается скачать артефакт датасета. В противном случае логирует конфигурационный файл датасета.

            :param data_file: Путь к конфигурационному файлу датасета.
            :return: Словарь с данными датасета.
            """
            with open(data_file) as f:
                data_config = yaml.safe_load(f)

            if data_config['path'].startswith(COMET_PREFIX):
                path = data_config['path'].replace(COMET_PREFIX, "")
                data_dict = self.download_dataset_artifact(path)

                return data_dict

            self.log_asset(self.opt.data, metadata={"type": "data-config-file"})

            return check_dataset(data_file)

        def log_predictions(self, image, labelsn, path, shape, predn):
            """
            Логирует предсказания модели для изображения. Логирует изображение и метаданные о детекциях,
            если они есть и если количество залогированных изображений не превышает максимально допустимое.

            :param image: Изображение (возможно, в виде тензора).
            :param labelsn: Соответствующие метки для изображения (возможно, в виде тензора).
            :param path: Путь к изображению.
            :param shape: Форма изображения (возможно, в виде кортежа).
            :param predn: Предсказания модели (возможно, в виде тензора).
            """
            if self.logged_images_count >= self.max_images:
                return
            detections = predn[predn[:, 4] > self.conf_thres]
            iou = box_iou(labelsn[:, 1:], detections[:, :4])
            mask, _ = torch.where(iou > self.iou_thres)
            if len(mask) == 0:
                return

            filtered_detections = detections[mask]
            filtered_labels = labelsn[mask]

            image_id = path.split("/")[-1].split(".")[0]
            image_name = f"{image_id}_curr_epoch_{self.experiment.curr_epoch}"
            if image_name not in self.logged_image_names:
                native_scale_image = Image.open(path)
                self.log_image(native_scale_image, name=image_name)
                self.logged_image_names.append(image_name)

            metadata = []
            for cls, *xyxy in filtered_labels.tolist():
                metadata.append({
                    "label": f"{self.class_names[int(cls)]}-gt",
                    "score": 100,
                    "box": {
                        "x": xyxy[0],
                        "y": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3]},
                })
            for *xyxy, conf, cls in filtered_detections.tolist():
                metadata.append({
                    "label": f"{self.class_names[int(cls)]}",
                    "score": conf * 100,
                    "box": {
                        "x": xyxy[0],
                        "y": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3]},
                })

            self.metadata_dict[image_name] = metadata
            self.logged_images_count += 1

            return

    def preprocess_prediction(self, image, labels, shape, pred):
        # Получаем количество меток и количество предсказаний
        nl, _ = labels.shape[0], pred.shape[0]

        # Предсказания
        # Если опция single_cls установлена, то устанавливаем все классы предсказаний равными 0 (один класс)
        if self.opt.single_cls:
            pred[:, 5] = 0

        # Создаем копию предсказаний, чтобы не изменять исходные данные
        predn = pred.clone()
        # Масштабируем ограничивающие рамки предсказаний до исходного размера изображения
        scale_boxes(image.shape[1:], predn[:, :4], shape[0], shape[1])

        labelsn = None
        # Если есть метки
        if nl:
            # Преобразуем координаты ограничивающих рамок меток из формата xywh в формат xyxy
            tbox = xywh2xyxy(labels[:, 1:5])  # ограничивающие рамки целей
            # Масштабируем ограничивающие рамки меток до исходного размера изображения
            scale_boxes(image.shape[1:], tbox, shape[0], shape[1])  # метки в исходном пространстве
            # Объединяем класс и ограничивающие рамки меток в одном тензоре
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # метки в исходном пространстве
            # Масштабируем ограничивающие рамки предсказаний до исходного размера изображения
            scale_boxes(image.shape[1:], predn[:, :4], shape[0], shape[1])  # предсказания в исходном пространстве

        return predn, labelsn

    def add_assets_to_artifact(self, artifact, path, asset_path, split):
        # Получаем список путей к изображениям, отсортированный по имени файла
        img_paths = sorted(glob.glob(f"{asset_path}/*"))
        # Получаем список путей к файлам меток, соответствующих изображениям
        label_paths = img2label_paths(img_paths)

        # Итерируемся по парам изображений и файлов меток
        for image_file, label_file in zip(img_paths, label_paths):
            # Получаем логические пути к изображению и файлу метки относительно заданного пути
            image_logical_path, label_logical_path = map(lambda x: os.path.relpath(x, path), [image_file, label_file])

            try:
                # Добавляем изображение в артефакт с указанием логического пути и метаданных о разбиении
                artifact.add(image_file, logical_path=image_logical_path, metadata={"split": split})
                # Добавляем файл метки в артефакт с указанием логического пути и метаданных о разбиении
                artifact.add(label_file, logical_path=label_logical_path, metadata={"split": split})
            except ValueError as e:
                # Логируем ошибку при добавлении файла в артефакт
                logger.error('COMET ERROR: Error adding file to Artifact. Skipping file.')
                # Логируем детали ошибки
                logger.error(f"COMET ERROR: {e}")
                # Пропускаем текущий файл и переходим к следующему
                continue

        return artifact

    def upload_dataset_artifact(self):
        # Получаем имя датасета из словаря данных, если его нет, то используем имя по умолчанию
        dataset_name = self.data_dict.get("dataset_name", "yolov5-dataset")
        # Получаем абсолютный путь к датасету
        path = str((ROOT / Path(self.data_dict["path"])).resolve())

        # Создаем копию словаря данных датасета для метаданных артефакта
        metadata = self.data_dict.copy()
        # Итерируемся по ключам, соответствующим разбиениям датасета (тренировочное, валидационное, тестовое)
        for key in ["train", "val", "test"]:
            # Получаем путь к разбиению датасета
            split_path = metadata.get(key)
            if split_path is not None:
                # Заменяем абсолютный путь на относительный в метаданных
                metadata[key] = split_path.replace(path, "")

        # Создаем артефакт Comet ML с указанным именем, типом и метаданными
        artifact = comet_ml.Artifact(name=dataset_name, artifact_type="dataset", metadata=metadata)
        # Итерируемся по ключам в метаданных
        for key in metadata.keys():
            if key in ["train", "val", "test"]:
                # Если параметр upload_dataset установлен как строка и не совпадает с текущим разбиением, то пропускаем
                if isinstance(self.upload_dataset, str) and (key != self.upload_dataset):
                    continue

                # Получаем путь к текущему разбиению датасета
                asset_path = self.data_dict.get(key)
                if asset_path is not None:
                    # Добавляем изображения и файлы меток текущего разбиения в артефакт
                    artifact = self.add_assets_to_artifact(artifact, path, asset_path, key)

        # Загружаем артефакт в эксперимент Comet ML
        self.experiment.log_artifact(artifact)

        return

    def download_dataset_artifact(self, artifact_path):
        # Получаем артефакт из эксперимента Comet ML по указанному пути
        logged_artifact = self.experiment.get_artifact(artifact_path)
        # Создаем путь для сохранения загруженного артефакта в директорию сохранения по умолчанию
        artifact_save_dir = str(Path(self.opt.save_dir) / logged_artifact.name)
        # Загружаем артефакт в созданную директорию
        logged_artifact.download(artifact_save_dir)

        # Получаем метаданные артефакта
        metadata = logged_artifact.metadata
        # Создаем копию метаданных для дальнейших изменений
        data_dict = metadata.copy()
        # Устанавливаем путь к датасету в словаре данных
        data_dict["path"] = artifact_save_dir

        # Получаем имена классов из метаданных
        metadata_names = metadata.get("names")
        # Если имена классов представлены в виде словаря
        if type(metadata_names) == dict:
            # Преобразуем их в словарь с ключами - целыми числами
            data_dict["names"] = {int(k): v for k, v in metadata.get("names").items()}
        # Если имена классов представлены в виде списка
        elif type(metadata_names) == list:
            # Создаем словарь с ключами - целыми числами, соответствующими индексам списка
            data_dict["names"] = {int(k): v for k, v in zip(range(len(metadata_names)), metadata_names)}
        else:
            # Если формат имен классов неверный, вызываем ошибку
            raise "Invalid 'names' field in dataset yaml file. Please use a list or dictionary"

        # Обновляем пути к данным в словаре данных
        data_dict = self.update_data_paths(data_dict)
        # Возвращаем обновленный словарь данных
        return data_dict

    def update_data_paths(self, data_dict):
        # Получаем путь к корневой директории датасета из словаря данных
        path = data_dict.get("path", "")

        # Итерируемся по ключам, соответствующим разбиениям датасета (тренировочное, валидационное, тестовое)
        for split in ["train", "val", "test"]:
            # Если путь к текущему разбиению датасета существует в словаре данных
            if data_dict.get(split):
                # Получаем путь к текущему разбиению
                split_path = data_dict.get(split)
                # Обновляем путь к текущему разбиению, добавляя корневой путь к датасету
                data_dict[split] = (f"{path}/{split_path}" if isinstance(split, str) else [
                    f"{path}/{x}" for x in split_path])

        # Возвращаем обновленный словарь данных с измененными путями
        return data_dict

    def on_pretrain_routine_end(self, paths):
        # Если,resume опция установлена, то выходим из функции без действий
        if self.opt.resume:
            return

        # Логируем каждый путь из списка путей
        for path in paths:
            self.log_asset(str(path))

        # Если флаг upload_dataset установлен
        if self.upload_dataset:
            # Если,resume опция не установлена
            if not self.resume:
                # Загружаем артефакт датасета
                self.upload_dataset_artifact()

        return

    def on_train_start(self):
        # Логируем гиперпараметры при начале обучения
        self.log_parameters(self.hyp)

    def on_train_epoch_start(self):
        # Ничего не делаем при начале каждой эпохи обучения
        return

    def on_train_epoch_end(self, epoch):
        # Устанавливаем текущую эпоху в эксперименте Comet ML
        self.experiment.curr_epoch = epoch

        return

    def on_train_batch_start(self):
        # Ничего не делаем при начале каждого батча обучения
        return

    def on_train_batch_end(self, log_dict, step):
        # Устанавливаем текущий шаг в эксперименте Comet ML
        self.experiment.curr_step = step
        # Если флаг log_batch_metrics установлен и текущий шаг кратный интервалу логирования батчей
        if self.log_batch_metrics and (step % self.comet_log_batch_interval == 0):
            # Логируем метрики для текущего батча
            self.log_metrics(log_dict, step=step)

        return

    def on_train_end(self, files, save_dir, last, best, epoch, results):
        # Если флаг логирования предсказаний в Comet установлен
        if self.comet_log_predictions:
            # Получаем текущую эпоху эксперимента Comet
            curr_epoch = self.experiment.curr_epoch
            # Логируем словарь метаданных изображений в виде JSON файла с указанием текущей эпохи
            self.experiment.log_asset_data(self.metadata_dict, "image-metadata.json", epoch=curr_epoch)

        # Логируем каждый файл из списка файлов с указанием эпохи в метаданных
        for f in files:
            self.log_asset(f, metadata={"epoch": epoch})
        # Логируем файл с результатами в формате CSV с указанием эпохи в метаданных
        self.log_asset(f"{save_dir}/results.csv", metadata={"epoch": epoch})

        # Если не выполняется эволюция модели (возможно, параметр evolve в опциях)
        if not self.opt.evolve:
            # Определяем путь к модели (используем лучшую модель, если она существует, иначе последнюю)
            model_path = str(best if best.exists() else last)
            # Получаем имя файла модели
            name = Path(model_path).name
            # Если флаг сохранения модели установлен
            if self.save_model:
                # Логируем модель в Comet с указанием имени модели, пути к файлу и других параметров
                self.experiment.log_model(
                    self.model_name,
                    file_or_folder=model_path,
                    file_name=name,
                    overwrite=True,
                )

        # Проверка, выполняется ли эксперимент с оптимизатором Comet
        if hasattr(self.opt, 'comet_optimizer_id'):
            # Получаем метрику, которая является объективом оптимизатора из результатов обучения
            metric = results.get(self.opt.comet_optimizer_metric)
            # Логируем значение метрики оптимизатора в Comet
            self.experiment.log_other('optimizer_metric_value', metric)

        # Завершаем текущий_run эксперимента Comet
        self.finish_run()

    def on_val_start(self):
        # Ничего не делаем при начале валидации
        return

    def on_val_batch_start(self):
        # Ничего не делаем при начале каждого батча валидации
        return

    def on_val_batch_end(self, batch_i, images, targets, paths, shapes, outputs):
        # Если флаг логирования предсказаний в Comet не установлен или текущий батч не соответствует интервалу логирования
        if not (self.comet_log_predictions and ((batch_i + 1) % self.comet_log_prediction_interval == 0)):
            return

        # Итерируемся по каждому выходу (предсказанию) в батче
        for si, pred in enumerate(outputs):
            # Если количество предсказаний равно 0, пропускаем текущий элемент
            if len(pred) == 0:
                continue

            # Получаем текущее изображение, метки, размер и путь
            image = images[si]
            labels = targets[targets[:, 0] == si, 1:]
            shape = shapes[si]
            path = paths[si]
            # Предобрабатываем предсказания и метки
            predn, labelsn = self.preprocess_prediction(image, labels, shape, pred)
            # Если есть обработанные метки
            if labelsn is not None:
                # Логируем предсказания и метки для текущего изображения
                self.log_predictions(image, labelsn, path, shape, predn)

        return

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        # Если флаг логирования метрик по классам в Comet установлен и количество классов больше 1
        if self.comet_log_per_class_metrics:
            if self.num_classes > 1:
                # Итерируемся по каждому классу и соответствующим метрикам
                for i, c in enumerate(ap_class):
                    # Получаем имя класса
                    class_name = self.class_names[c]
                    # Логируем метрики для текущего класса в Comet с указанием префикса (имени класса)
                    self.experiment.log_metrics(
                        {
                            'mAP@.5': ap50[i],
                            'mAP@.5:.95': ap[i],
                            'precision': p[i],
                            'recall': r[i],
                            'f1': f1[i],
                            'true_positives': tp[i],
                            'false_positives': fp[i],
                            'support': nt[c]},
                        prefix=class_name)

        if self.comet_log_confusion_matrix:
            # Получаем номер текущей эпохи из эксперимента Comet
            epoch = self.experiment.curr_epoch
            # Получаем список имен классов из словаря имен классов
            class_names = list(self.class_names.values())
            # Добавляем класс "background" в список имен классов
            class_names.append("background")
            # Получаем количество классов (включая "background")
            num_classes = len(class_names)

            # Логируем матрицу ошибок в эксперименте Comet
            self.experiment.log_confusion_matrix(
                # Матрица ошибок
                matrix=confusion_matrix.matrix,
                # Максимальное количество категорий для отображения
                max_categories=num_classes,
                # Метки классов
                labels=class_names,
                # Номер эпохи
                epoch=epoch,
                # Название столбцов в матрице ошибок (реальные категории)
                column_label='Actual Category',
                # Название строк в матрице ошибок (предсказанные категории)
                row_label='Predicted Category',
                # Имя файла для сохранения матрицы ошибок
                file_name=f"confusion-matrix-epoch-{epoch}.json",
            )

        def on_fit_epoch_end(self, result, epoch):
            # Логируем метрики, полученные в конце эпохи обучения, указав номер эпохи
            self.log_metrics(result, epoch=epoch)

        def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
            # Проверяем, нужно ли сохранять модель в текущей эпохе
            # Сохраняем модель, если текущая эпоха кратна периоду сохранения и это не последняя эпоха, и период сохранения не равен -1
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                # Логируем модель в эксперименте Comet
                self.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

        def on_params_update(self, params):
            # Логируем обновленные параметры в эксперименте Comet
            self.log_parameters(params)

        def finish_run(self):
            # Завершаем эксперимент Comet
            self.experiment.end()