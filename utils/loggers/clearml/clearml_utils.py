"""Main Logger class for ClearML experiment tracking."""
import glob
import re
from pathlib import Path

import numpy as np
import yaml

from utils.plots import Annotator, colors

try:
    # Попытка импорта модуля clearml и его классов Dataset, Task
    import clearml
    from clearml import Dataset, Task

    # Проверка, что clearml импортирован как пакет, а не из локальной директории
    assert hasattr(clearml, '__version__')
except (ImportError, AssertionError):
    # Если произошла ошибка импорта, clearml остается None
    clearml = None


def construct_dataset(clearml_info_string):
    # Извлечение идентификатора набора данных из строки clearml_info_string
    dataset_id = clearml_info_string.replace('clearml://', '')
    # Получение объекта набора данных по идентификатору
    dataset = Dataset.get(dataset_id=dataset_id)
    # Получение локального пути к набору данных
    dataset_root_path = Path(dataset.get_local_copy())

    # Поиск файлов с расширениями.yaml и.yml в корневой директории набора данных
    yaml_filenames = list(glob.glob(str(dataset_root_path / "*.yaml")) + glob.glob(str(dataset_root_path / "*.yml")))
    if len(yaml_filenames) > 1:
        # Если найдено более одного файла.yaml, вызывается ошибка
        raise ValueError('More than one yaml file was found in the dataset root, cannot determine which one contains the dataset definition this way.')
    elif len(yaml_filenames) == 0:
        # Если не найдено ни одного файла.yaml, вызывается ошибка
        raise ValueError('No yaml definition found in dataset root path, check that there is a correct yaml file inside the dataset root path.')
    with open(yaml_filenames[0]) as f:
        # Загрузка содержимого файла.yaml в словарь
        dataset_definition = yaml.safe_load(f)

    # Проверка, что словарь dataset_definition содержит все необходимые ключи
    assert set(dataset_definition.keys()).issuperset(
        {'train', 'test', 'val', 'nc', 'names'}
    ), "The right keys were not found in the yaml file, make sure it at least has the following keys: ('train', 'test', 'val', 'nc', 'names')"

    # Инициализация словаря для хранения данных набора
    data_dict = dict()
    data_dict['train'] = str(
        (dataset_root_path / dataset_definition['train']).resolve()) if dataset_definition['train'] else None
    data_dict['test'] = str(
        (dataset_root_path / dataset_definition['test']).resolve()) if dataset_definition['test'] else None
    data_dict['val'] = str(
        (dataset_root_path / dataset_definition['val']).resolve()) if dataset_definition['val'] else None
    data_dict['nc'] = dataset_definition['nc']
    data_dict['names'] = dataset_definition['names']

    return data_dict


class ClearmlLogger:
    def __init__(self, opt, hyp):
        # Текущий номер эпохи
        self.current_epoch = 0
        # Множество для отслеживания уже залогированных изображений в текущей эпохе
        self.current_epoch_logged_images = set()
        # Максимальное количество изображений, которые можно залогировать в ClearML за эпоху
        self.max_imgs_to_log_per_epoch = 16
        # Интервал эпох, с которым должны логироваться изображения с ограничивающими рамками
        self.bbox_interval = opt.bbox_interval
        # Ссылка на модуль clearml
        self.clearml = clearml
        # Объект ClearML Task для отслеживания эксперимента
        self.task = None
        # Словарь с данными набора
        self.data_dict = None
        if self.clearml:
            # Инициализация задачи ClearML
            self.task = Task.init(
                project_name=opt.project if opt.project != 'runs/train' else 'YOLOv5',
                task_name=opt.name if opt.name != 'exp' else 'Training',
                tags=['YOLOv5'],
                output_uri=True,
                auto_connect_frameworks={'pytorch': False}
            )
            # Подключение гиперпараметров к задаче ClearML
            self.task.connect(hyp, name='Hyperparameters')

            # Получение версии набора данных ClearML, если требуется
            if opt.data.startswith('clearml://'):
                self.data_dict = construct_dataset(opt.data)
                opt.data = self.data_dict

    def log_debug_samples(self, files, title='Debug Samples'):
        # Логирование файлов (изображений) как отладочных примеров в задаче ClearML
        for f in files:
            if f.exists():
                it = re.search(r'_batch(\d+)', f.name)
                iteration = int(it.groups()[0]) if it else 0
                self.task.get_logger().report_image(title=title,
                                                    series=f.name.replace(it.group(), ''),
                                                    local_path=str(f),
                                                    iteration=iteration)

    def log_image_with_boxes(self, image_path, boxes, class_names, image, conf_threshold=0.25):
        if len(self.current_epoch_logged_images) < self.max_imgs_to_log_per_epoch and self.current_epoch >= 0:
            if self.current_epoch % self.bbox_interval == 0 and image_path not in self.current_epoch_logged_images:
                # Преобразование тензора изображения в массив numpy
                im = np.ascontiguousarray(np.moveaxis(image.mul(255).clamp(0, 255).byte().cpu().numpy(), 0, 2))
                # Создание объекта Annotator для рисования на изображении
                annotator = Annotator(im=im, pil=True)
                for i, (conf, class_nr, box) in enumerate(zip(boxes[:, 4], boxes[:, 5], boxes[:, :4])):
                    # Получение цвета для ограничивающей рамки
                    color = colors(i)

                    # Получение имени класса по номеру класса
                    class_name = class_names[int(class_nr)]
                    # Вычисление процента уверенности
                    confidence_percentage = round(float(conf) * 100, 2)
                    # Формирование метки для ограничивающей рамки
                    label = f"{class_name}: {confidence_percentage}%"

                    if conf > conf_threshold:
                        # Рисование ограничивающей рамки на изображении
                        annotator.rectangle(box.cpu().numpy(), outline=color)
                        # Добавление метки к ограничивающей рамке
                        annotator.box_label(box.cpu().numpy(), label=label, color=color)

                # Получение изображения с нарисованными ограничивающими рамками
                annotated_image = annotator.result()
                # Отправка изображения с ограничивающими рамками в ClearML
                self.task.get_logger().report_image(title='Bounding Boxes',
                                                    series=image_path.name,
                                                    iteration=self.current_epoch,
                                                    image=annotated_image)
                # Добавление пути к изображению в множество уже залогированных изображений
                self.current_epoch_logged_images.add(image_path)