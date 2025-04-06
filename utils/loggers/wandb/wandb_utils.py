"""Utilities and tools for tracking runs with Weights & Biases."""

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # Корневая директория YOLOv5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Добавление корневой директории в системный путь

from utils.dataloaders import LoadImagesAndLabels, img2label_paths
from utils.general import LOGGER, check_dataset, check_file

try:
    import wandb

    assert hasattr(wandb, '__version__')  # Проверка импорта пакета, а не локального каталога
except (ImportError, AssertionError):
    wandb = None

RANK = int(os.getenv('RANK', -1))
WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def remove_prefix(from_string, prefix=WANDB_ARTIFACT_PREFIX):
    return from_string[len(prefix):]


def check_wandb_config_file(data_config_file):
    wandb_config = '_wandb.'.join(data_config_file.rsplit('.', 1))  # Обновленный путь к data.yaml
    return wandb_config if Path(wandb_config).is_file() else data_config_file


def check_wandb_dataset(data_file):
    if isinstance(data_file, dict):
        return data_file  # Другой менеджер датасета уже обработал данные
    if check_file(data_file) and data_file.endswith('.yaml'):
        with open(data_file, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
        is_train_wandb = isinstance(data_dict.get('train'), str) and data_dict['train'].startswith(
            WANDB_ARTIFACT_PREFIX)
        is_val_wandb = isinstance(data_dict.get('val'), str) and data_dict['val'].startswith(WANDB_ARTIFACT_PREFIX)
        return data_dict if (is_train_wandb or is_val_wandb) else check_dataset(data_file)
    return check_dataset(data_file)


def get_run_info(run_path):
    run_path = Path(remove_prefix(run_path))
    return (run_path.parent.parent.stem, run_path.parent.stem, run_path.stem,
            f'run_{run_path.stem}_model')


def check_wandb_resume(opt):
    if RANK not in [-1, 0]:
        process_wandb_config_ddp_mode(opt)
    if isinstance(opt.resume, str) and opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
        if RANK not in [-1, 0]:  # Для возобновления DDP-сессий
            entity, project, run_id, model_artifact_name = get_run_info(opt.resume)
            api = wandb.Api()
            artifact = api.artifact(f"{entity}/{project}/{model_artifact_name}:latest")
            opt.weights = str(Path(artifact.download()) / "last.pt")
        return True
    return None


def process_wandb_config_ddp_mode(opt):
    with open(check_file(opt.data), errors='ignore') as f:
        data_dict = yaml.safe_load(f)

    train_dir, val_dir = None, None
    if isinstance(data_dict.get('train'), str) and data_dict['train'].startswith(WANDB_ARTIFACT_PREFIX):
        api = wandb.Api()
        train_artifact = api.artifact(remove_prefix(data_dict['train']) + f':{opt.artifact_alias}')
        train_dir = train_artifact.download()
        data_dict['train'] = str(Path(train_dir) / 'data/images/')

    if isinstance(data_dict.get('val'), str) and data_dict['val'].startswith(WANDB_ARTIFACT_PREFIX):
        api = wandb.Api()
        val_artifact = api.artifact(remove_prefix(data_dict['val']) + f':{opt.artifact_alias}')
        val_dir = val_artifact.download()
        data_dict['val'] = str(Path(val_dir) / 'data/images/')

    if train_dir or val_dir:
        ddp_data_path = str(Path(val_dir) / 'wandb_local_data.yaml')
        with open(ddp_data_path, 'w') as f:
            yaml.safe_dump(data_dict, f)
        opt.data = ddp_data_path


class WandbLogger():
    """Логирует тренировочные сессии, датасеты, модели и предсказания в Weights & Biases.

    Этот логгер отправляет информацию на платформу wandb.ai. По умолчанию
    информация включает гиперпараметры, системные настройки, метрики модели
    и базовые данные. Дополнительные аргументы командной строки позволяют
    залогировать датасеты, модели и предсказания.

    Подробнее в документации Weights & Biases:
    https://docs.wandb.com/guides/integrations/yolov5
    """

    def __init__(self, opt, run_id=None, job_type='Training'):
        """
        Инициализация WandbLogger:
        - Загружает датасет, если указан флаг opt.upload_dataset
        - Настраивает тренировочный процесс для job_type='Training'

        Аргументы:
        opt (namespace) -- Аргументы командной строки
        run_id (str) -- ID сессии W&B для возобновления
        job_type (str) -- Тип задания (Training/Dataset Creation)
        """
        if opt.upload_dataset:
            opt.upload_dataset = False  # Временное отключение из-за ошибки

        self.job_type = job_type
        self.wandb, self.wandb_run = wandb, None
        self.val_artifact = self.train_artifact = None
        self.max_imgs_to_log = 16
        self.data_dict = None

        if isinstance(opt.resume, str) and opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            entity, project, run_id, model_artifact_name = get_run_info(opt.resume)
            self.wandb_run = wandb.init(id=run_id, project=project, entity=entity,
                                        resume='allow', allow_val_change=True)
            opt.resume = model_artifact_name
        elif self.wandb:
            project_name = Path(opt.project).stem if opt.project != 'runs/train' else 'YOLOv5'
            self.wandb_run = wandb.init(config=opt, resume="allow", project=project_name,
                                        entity=opt.entity, name=opt.name if opt.name != 'exp' else None,
                                        job_type=job_type, id=run_id, allow_val_change=True)

        if self.wandb_run:
            if self.job_type == 'Training':
                if opt.upload_dataset and not opt.resume:
                    self.wandb_artifact_data_dict = self.check_and_upload_dataset(opt)

                self.data_dict = opt.data if isinstance(opt.data, dict) else check_wandb_dataset(opt.data)
                if not opt.resume:
                    self.wandb_run.config.update({'data_dict': self.data_dict}, allow_val_change=True)
                self.setup_training(opt)

            if self.job_type == 'Dataset Creation':
                self.wandb_run.config.update({"upload_dataset": True})
                self.data_dict = self.check_and_upload_dataset(opt)

    def check_and_upload_dataset(self, opt):
        """
        Проверяет совместимость датасета и загружает его как артефакт W&B

        Аргументы:
        opt (namespace) -- Аргументы командной строки текущего запуска

        Возвращает:
        Обновленный словарь с данными датасета, где локальные пути заменены ссылками на артефакты
        """
        assert wandb, 'Установите wandb для загрузки датасета'
        # Определение имени проекта для артефакта
        project_name = 'YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem
        # Загрузка датасета в W&B и получение пути к конфигурационному файлу
        config_path = self.log_dataset_artifact(opt.data, opt.single_cls, project_name)
        # Чтение обновленного конфигурационного файла
        with open(config_path, errors='ignore') as f:
            wandb_data_dict = yaml.safe_load(f)
        return wandb_data_dict

    def setup_training(self, opt):
        """
        Настройка процесса обучения YOLO:
          - Загрузка чекпоинта модели и датасета при resume из артефакта
          - Обновление data_dict с информацией о предыдущем запуске
          - Инициализация таблиц для логов и интервала логирования bounding box

        Аргументы:
        opt (namespace) -- Аргументы командной строки текущего запуска
        """
        self.log_dict, self.current_epoch = {}, 0
        self.bbox_interval = opt.bbox_interval

        # Если происходит resume из артефакта
        if isinstance(opt.resume, str):
            modeldir, _ = self.download_model_artifact(opt)
            if modeldir:
                self.weights = Path(modeldir) / "last.pt"
                config = self.wandb_run.config
                # Обновление параметров обучения из конфигурации W&B
                opt.weights, opt.save_period, opt.batch_size, opt.bbox_interval, opt.epochs, opt.hyp, opt.imgsz = \
                    str(self.weights), config.save_period, config.batch_size, config.bbox_interval, config.epochs, \
                        config.hyp, config.imgsz

        data_dict = self.data_dict
        # Загрузка датасета из артефакта при необходимости
        if self.val_artifact is None:
            self.train_artifact_path, self.train_artifact = self.download_dataset_artifact(
                data_dict.get('train'), opt.artifact_alias)
            self.val_artifact_path, self.val_artifact = self.download_dataset_artifact(
                data_dict.get('val'), opt.artifact_alias)

        # Обновление путей в data_dict на локальные
        if self.train_artifact_path:
            data_dict['train'] = str(Path(self.train_artifact_path) / 'data/images/')
        if self.val_artifact_path:
            data_dict['val'] = str(Path(self.val_artifact_path) / 'data/images/')

        # Инициализация таблиц для отображения результатов
        if self.val_artifact:
            self.result_artifact = wandb.Artifact(f"run_{wandb.run.id}_progress", "evaluation")
            columns = ["epoch", "id", "ground truth", "prediction"] + self.data_dict['names']
            self.result_table = wandb.Table(columns)
            self.val_table = self.val_artifact.get("val")
            if not self.val_table_path_map:
                self.map_val_table_path()

        # Автоматическое определение интервала логирования bounding box
        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else 1
            if opt.evolve or opt.noplots:
                self.bbox_interval = opt.bbox_interval = opt.epochs + 1  # Отключение логирования

        # Обновление data_dict для использования локальных артефактов
        if self.train_artifact_path and self.val_artifact_path:
            self.data_dict = data_dict

    def download_dataset_artifact(self, path, alias):
        """
        Загружает датасет из артефакта W&B, если путь начинается с WANDB_ARTIFACT_PREFIX

        Аргументы:
        path -- Путь к датасету
        alias (str) -- псевдоним артефакта для загрузки

        Возвращает:
        (str, wandb.Artifact) -- Путь к загруженному датасету и объект артефакта
        """
        if isinstance(path, str) and path.startswith(WANDB_ARTIFACT_PREFIX):
            artifact_path = Path(remove_prefix(path) + f":{alias}")
            dataset_artifact = wandb.use_artifact(artifact_path.as_posix().replace("\\", "/"))
            assert dataset_artifact, "Ошибка: артефакт датасета не найден"
            datadir = dataset_artifact.download()
            return datadir, dataset_artifact
        return None, None

    def download_model_artifact(self, opt):
        """
        Загружает чекпоинт модели из артефакта W&B, если путь resume начинается с WANDB_ARTIFACT_PREFIX

        Аргументы:
        opt (namespace) -- Аргументы командной строки текущего запуска
        """
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            # Получаем имя артефакта модели без префикса
            artifact_name = remove_prefix(opt.resume, WANDB_ARTIFACT_PREFIX)
            # Загружаем последнюю версию артефакта
            model_artifact = wandb.use_artifact(f"{artifact_name}:latest")
            assert model_artifact, 'Ошибка: артефакт модели не найден'
            # Сохраняем модель локально
            modeldir = model_artifact.download()
            # Проверяем, завершена ли тренировка
            total_epochs = model_artifact.metadata.get('total_epochs')
            if total_epochs is not None:
                raise AssertionError('Тренировка завершена. Можно возобновлять только незавершенные сессии.')
            return modeldir, model_artifact
        return None, None

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        """
        Логирует чекпоинт модели в W&B как артефакт

        Аргументы:
        path (Path) -- Путь к директории с чекпоинтами
        opt (namespace) -- Аргументы командной строки
        epoch (int) -- Текущая эпоха обучения
        fitness_score (float) -- Значение метрики качества
        best_model (bool) -- Лучшая модель на текущий момент
        """
        # Создаем артефакт модели с метаданными
        model_artifact = wandb.Artifact(
            f'run_{wandb.run.id}_model',
            type='model',
            metadata={
                'original_url': str(path),
                'epochs_trained': epoch + 1,
                'save_period': opt.save_period,
                'project': opt.project,
                'total_epochs': opt.epochs,
                'fitness_score': fitness_score
            }
        )
        # Добавляем файл last.pt в артефакт
        model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
        # Логируем артефакт с соответствующими псевдонимами
        wandb.log_artifact(
            model_artifact,
            aliases=['latest', 'last', f'epoch {epoch + 1}', 'best' if best_model else '']
        )
        LOGGER.info(f"Сохранение модели на эпохе {epoch + 1}")

    def log_dataset_artifact(self, data_file, single_cls, project, overwrite_config=False):
        """
        Логирует датасет в W&B как артефакт и возвращает новый .yaml с ссылками на артефакты

        Аргументы:
        data_file (str) -- Путь к .yaml с информацией о датасете
        single_cls (bool) -- Обрабатывать как одноклассовый датасет
        project (str) -- Название проекта
        overwrite_config (bool) -- Перезаписать исходный .yaml или создать новый

        Возвращает:
        str -- Путь к новому .yaml с ссылками на артефакты
        """
        upload_dataset = self.wandb_run.config.upload_dataset
        log_val_only = isinstance(upload_dataset, str) and upload_dataset == 'val'

        # Парсим и проверяем датасет
        self.data_dict = check_dataset(data_file)
        data = dict(self.data_dict)
        # Обрабатываем классы
        nc = 1 if single_cls else int(data['nc'])
        names = ['item'] if single_cls else data['names']
        names = {k: v for k, v in enumerate(names)}

        # Логируем тренировочный набор
        if not log_val_only and data.get('train'):
            self.train_artifact = self.create_dataset_table(
                LoadImagesAndLabels(data['train'], rect=True, batch_size=1),
                names,
                name='train'
            )
            data['train'] = f"{WANDB_ARTIFACT_PREFIX}{Path(project) / 'train'}"

        # Логируем валидационный набор
        if data.get('val'):
            self.val_artifact = self.create_dataset_table(
                LoadImagesAndLabels(data['val'], rect=True, batch_size=1),
                names,
                name='val'
            )
            data['val'] = f"{WANDB_ARTIFACT_PREFIX}{Path(project) / 'val'}"

        # Создаем новый .yaml с ссылками на артефакты
        path = Path(data_file)
        filename = path.stem + ('_wandb' if not overwrite_config else '') + '.yaml'
        new_path = ROOT / 'data' / filename
        with open(new_path, 'w') as f:
            data.pop('download', None)
            data.pop('path', None)
            yaml.safe_dump(data, f)
            LOGGER.info(f"Создан конфигурационный файл датасета: {new_path}")

        # Логируем артефакты в W&B
        if self.job_type == 'Training':
            if not log_val_only:
                self.wandb_run.log_artifact(self.train_artifact)
            self.wandb_run.use_artifact(self.val_artifact)
            self.val_artifact.wait()
            self.val_table = self.val_artifact.get('val')
            self.map_val_table_path()
        else:
            self.wandb_run.log_artifact(self.train_artifact)
            self.wandb_run.log_artifact(self.val_artifact)

        return new_path

    def map_val_table_path(self):
        """
        Сопоставляет имена файлов из валидационного датасета с их идентификаторами в таблице W&B.
        Полезно для ссылки на артефакты при оценке модели.
        """
        self.val_table_path_map = {}
        LOGGER.info("Сопоставление датасета")
        # Проходим по данным валидационной таблицы
        for i, data in enumerate(tqdm(self.val_table.data)):
            # Сопоставляем имя файла (находится в data[3]) с его идентификатором (находится в data[0])
            self.val_table_path_map[data[3]] = data[0]

    def create_dataset_table(self, dataset: LoadImagesAndLabels, class_to_id: Dict[int, str], name: str = 'dataset'):
        """
        Создает и возвращает артефакт W&B, содержащий таблицу W&B для датасета.

        Аргументы:
        dataset -- экземпляр класса LoadImagesAndLabels, используемый для итерации по данным для построения таблицы
        class_to_id -- хэш - таблица, которая сопоставляет идентификаторы классов с метками
        name -- имя артефакта

        Возвращает:
        Артефакт датасета, который можно залогировать или использовать
        """
        # TODO: Исследовать возможность использования мультипроцессинга для параллельного выполнения этого цикла. Это необходимо для ускорения логирования
        # Создаем артефакт с указанным именем и типом "dataset"
        artifact = wandb.Artifact(name=name, type="dataset")
        # Определяем список файлов изображений для обработки
        img_files = tqdm([dataset.path]) if isinstance(dataset.path, str) and Path(dataset.path).is_dir() else None
        img_files = tqdm(dataset.im_files) if not img_files else img_files
        for img_file in img_files:
            if Path(img_file).is_dir():
                # Если путь является директорией, добавляем ее в артефакт
                artifact.add_dir(img_file, name='data/images')
                # Определяем путь к файлам меток
                labels_path = 'labels'.join(dataset.path.rsplit('images', 1))
                # Добавляем директорию с метками в артефакт
                artifact.add_dir(labels_path, name='data/labels')
            else:
                # Если путь является файлом, добавляем его в артефакт
                artifact.add_file(img_file, name='data/images/' + Path(img_file).name)
                # Получаем путь к файлу метки для данного изображения
                label_file = Path(img2label_paths([img_file])[0])
                if label_file.exists():
                    # Если файл метки существует, добавляем его в артефакт
                    artifact.add_file(str(label_file), name='data/labels/' + label_file.name)
        # Создаем таблицу W&B с указанными колонками
        table = wandb.Table(columns=["id", "train_image", "Classes", "name"])
        # Создаем набор классов для W&B
        class_set = wandb.Classes([{'id': id, 'name': name} for id, name in class_to_id.items()])
        for si, (img, labels, paths, shapes) in enumerate(tqdm(dataset)):
            box_data, img_classes = [], {}
            for cls, *xywh in labels[:, 1:].tolist():
                cls = int(cls)
                # Создаем данные о ограничивающей рамке
                box_data.append({
                    "position": {
                        "middle": [xywh[0], xywh[1]],
                        "width": xywh[2],
                        "height": xywh[3]
                    },
                    "class_id": cls,
                    "box_caption": "%s" % (class_to_id[cls])
                })
                img_classes[cls] = class_to_id[cls]
            # Создаем словарь с данными о ограничивающих рамках для вывода
            boxes = {"ground_truth": {"box_data": box_data, "class_labels": class_to_id}}
            # Добавляем строку в таблицу
            table.add_data(si, wandb.Image(paths, classes=class_set, boxes=boxes), list(img_classes.values()),
                           Path(paths).name)
        # Добавляем таблицу в артефакт
        artifact.add(table, name)
        return artifact

    def log_training_progress(self, predn, path, names):
        """
        Создает таблицу оценки. Использует ссылки на валидационную таблицу датасета.

        Аргументы:
        predn (list): Список предсказаний в оригинальном масштабе [xmin, ymin, xmax, ymax, уверенность, класс]
        path (str): Локальный путь к текущему изображению
        names (dict(int, str)): Карта соответствия ID классов их названиям
        """
        class_set = wandb.Classes([{'id': id, 'name': name} for id, name in names.items()])
        box_data = []
        avg_conf_per_class = [0] * len(self.data_dict['names'])
        pred_class_count = {}
        for *xyxy, conf, cls in predn.tolist():
            if conf >= 0.25:
                cls = int(cls)
                box_data.append({
                    "position": {
                        "minX": xyxy[0],
                        "minY": xyxy[1],
                        "maxX": xyxy[2],
                        "maxY": xyxy[3]
                    },
                    "class_id": cls,
                    "box_caption": f"{names[cls]} {conf:.3f}",
                    "scores": {"class_score": conf},
                    "domain": "pixel"
                })
                avg_conf_per_class[cls] += conf
                pred_class_count[cls] = pred_class_count.get(cls, 0) + 1

        for cls in pred_class_count:
            avg_conf_per_class[cls] /= pred_class_count[cls]

        boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
        id = self.val_table_path_map[Path(path).name]
        self.result_table.add_data(
            self.current_epoch,
            id,
            self.val_table.data[id][1],
            wandb.Image(self.val_table.data[id][1], boxes=boxes, classes=class_set),
            *avg_conf_per_class
        )

    def val_one_image(self, pred, predn, path, names, im):
        """
        Логирует данные валидации для одного изображения. Обновляет таблицу результатов и панель BoundingBox.

        Аргументы:
        pred (list): Масштабированные предсказания [xmin, ymin, xmax, ymax, уверенность, класс]
        predn (list): Предсказания в оригинальном масштабе
        path (str): Локальный путь к изображению
        names (dict): Карта классов
        im: Исходное изображение
        """
        if self.val_table and self.result_table:
            self.log_training_progress(predn, path, names)

        if len(self.bbox_media_panel_images) < self.max_imgs_to_log and self.current_epoch > 0:
            if self.current_epoch % self.bbox_interval == 0:
                box_data = [{
                    "position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                    "class_id": int(cls),
                    "box_caption": f"{names[int(cls)]} {conf:.3f}",
                    "scores": {"class_score": conf},
                    "domain": "pixel"
                } for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                self.bbox_media_panel_images.append(wandb.Image(im, boxes=boxes, caption=path.name))

    def log(self, log_dict):
        """
        Сохраняет метрики в буфер для логирования.

        Аргументы:
        log_dict (Dict): Метрики/медиа для записи
        """
        if self.wandb_run:
            self.log_dict.update(log_dict)

    def end_epoch(self, best_result=False):
        """
        Сохраняет логи, артефакты и таблицы в W&B. Сбрасывает буфер.

        Аргументы:
        best_result (bool): Лучший результат на текущей эпохе
        """
        if self.wandb_run:
            with all_logging_disabled():
                if self.bbox_media_panel_images:
                    self.log_dict["BoundingBoxDebugger"] = self.bbox_media_panel_images
                try:
                    wandb.log(self.log_dict)
                except BaseException as e:
                    LOGGER.info(f"Ошибка в W&B логгере: {e}")
                    self.wandb_run.finish()
                    self.wandb_run = None
                self.log_dict = {}
                self.bbox_media_panel_images = []

            if self.result_artifact:
                self.result_artifact.add(self.result_table, 'result')
                wandb.log_artifact(
                    self.result_artifact,
                    aliases=['latest', 'last', f'epoch {self.current_epoch}', 'best' if best_result else '']
                )
                wandb.log({"evaluation": self.result_table})
                columns = ["epoch", "id", "ground truth", "prediction"] + self.data_dict['names']
                self.result_table = wandb.Table(columns)
                self.result_artifact = wandb.Artifact(f"run_{wandb.run.id}_progress", "evaluation")

    def finish_run(self):
        """
        Завершает работу с W&B: сохраняет оставшиеся логи и завершает сессию
        """
        if self.wandb_run:
            if self.log_dict:
                with all_logging_disabled():
                    wandb.log(self.log_dict)
            wandb.run.finish()

    @contextmanager
    def all_logging_disabled(highest_level=logging.CRITICAL):
        """
        Контекстный менеджер для временного отключения логирования.
        Источник: https://gist.github.com/simon-weber/7853144
        """
        previous_level = logging.root.manager.disable
        logging.disable(highest_level)
        try:
            yield
        finally:
            logging.disable(previous_level)