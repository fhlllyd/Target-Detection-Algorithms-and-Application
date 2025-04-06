# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Logging utils
"""

import os
import warnings
from pathlib import Path

import pkg_resources as pkg
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.general import LOGGER, colorstr, cv2
from utils.loggers.clearml.clearml_utils import ClearmlLogger
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.plots import plot_images, plot_labels, plot_results
from utils.torch_utils import de_parallel

# Определение поддерживаемых логгеров в виде кортежа. Включает CSV, TensorBoard, Weights & Biases, ClearML и Comet.
LOGGERS = ('csv', 'tb', 'wandb', 'clearml', 'comet')  # *.csv, TensorBoard, Weights & Biases, ClearML
# Получение значения переменной окружения 'RANK'. Если переменная не установлена, используется значение -1.
RANK = int(os.getenv('RANK', -1))

try:
    # Попытка импорта библиотеки wandb.
    import wandb
    # Проверка, что импортирована библиотека, а не локальная директория с таким именем.
    assert hasattr(wandb, '__version__')
    # Проверка версии библиотеки wandb и текущего ранга.
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2') and RANK in {0, -1}:
        try:
            # Попытка авторизации в wandb с таймаутом 30 секунд.
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            # Обработка ошибки авторизации в случае, если терминал не поддерживает TTY.
            wandb_login_success = False
        if not wandb_login_success:
            # Если авторизация не удалась, устанавливаем wandb в None.
            wandb = None
except (ImportError, AssertionError):
    # Если произошла ошибка импорта или утверждение не выполнено, устанавливаем wandb в None.
    wandb = None

try:
    # Попытка импорта библиотеки clearml.
    import clearml
    # Проверка, что импортирована библиотека, а не локальная директория с таким именем.
    assert hasattr(clearml, '__version__')
except (ImportError, AssertionError):
    # Если произошла ошибка импорта или утверждение не выполнено, устанавливаем clearml в None.
    clearml = None

try:
    # Проверка текущего ранга. Если ранг не 0 или -1, comet_ml не используется.
    if RANK not in [0, -1]:
        comet_ml = None
    else:
        # Попытка импорта библиотеки comet_ml.
        import comet_ml
        # Проверка, что импортирована библиотека, а не локальная директория с таким именем.
        assert hasattr(comet_ml, '__version__')
        # Импорт класса CometLogger из модуля utils.loggers.comet.
        from utils.loggers.comet import CometLogger

except (ModuleNotFoundError, ImportError, AssertionError):
    # Если произошла ошибка импорта или утверждение не выполнено, устанавливаем comet_ml в None.
    comet_ml = None


class Loggers():
    # Класс Loggers для YOLOv5, который управляет различными логгерами.
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        # Директория для сохранения логов.
        self.save_dir = save_dir
        # Веса модели.
        self.weights = weights
        # Опции командной строки.
        self.opt = opt
        # Гиперпараметры модели.
        self.hyp = hyp
        # Флаг, указывающий, нужно ли создавать графики результатов.
        self.plots = not opt.noplots  # plot results
        # Логгер для вывода результатов в консоль.
        self.logger = logger
        # Список поддерживаемых логгеров.
        self.include = include
        # Список ключей для логгирования различных метрик и потерь.
        self.keys = [
            'train/box_loss',
            'train/obj_loss',
            'train/cls_loss',  # train loss
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',  # metrics
            'val/box_loss',
            'val/obj_loss',
            'val/cls_loss',  # val loss
            'x/lr0',
            'x/lr1',
            'x/lr2']  # params
        # Список ключей для логгирования лучших результатов.
        self.best_keys = ['best/epoch', 'best/precision', 'best/recall', 'best/mAP_0.5', 'best/mAP_0.5:0.95']
        for k in LOGGERS:
            # Инициализация пустых словарей для каждого логгера.
            setattr(self, k, None)
        # Флаг, указывающий, что всегда нужно логировать в CSV.
        self.csv = True

        # Сообщения
        # if not wandb:
        #     prefix = colorstr('Weights & Biases: ')
        #     s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs in Weights & Biases"
        #     self.logger.info(s)
        if not clearml:
            # Префикс для сообщения о ClearML.
            prefix = colorstr('ClearML: ')
            # Сообщение о необходимости установить ClearML для отслеживания и визуализации экспериментов.
            s = f"{prefix}run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 in ClearML"
            # Вывод сообщения в логгер.
            self.logger.info(s)
        if not comet_ml:
            # Префикс для сообщения о Comet.
            prefix = colorstr('Comet: ')
            # Сообщение о необходимости установить Comet для отслеживания и визуализации экспериментов.
            s = f"{prefix}run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet"
            # Вывод сообщения в логгер.
            self.logger.info(s)
        # TensorBoard
        # Получение директории для сохранения логов TensorBoard.
        s = self.save_dir
        if 'tb' in self.include and not self.opt.evolve:
            # Префикс для сообщения о TensorBoard.
            prefix = colorstr('TensorBoard: ')
            # Сообщение о запуске TensorBoard и просмотре результатов в браузере.
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            # Инициализация объекта SummaryWriter для TensorBoard.
            self.tb = SummaryWriter(str(s))

            # W&B
            # Проверка, что библиотека wandb импортирована и 'wandb' включен в список поддерживаемых логгеров
            if wandb and 'wandb' in self.include:
                # Проверка, если параметр resume - это строка и начинается с 'wandb - artifact://'
                wandb_artifact_resume = isinstance(self.opt.resume, str) and self.opt.resume.startswith(
                    'wandb-artifact://')
                # Получение идентификатора запуска Wandb из сохраненных весов модели, если есть параметр resume и не используется возобновление по артефакту Wandb
                run_id = torch.load(self.weights).get(
                    'wandb_id') if self.opt.resume and not wandb_artifact_resume else None
                # Добавление гиперпараметров в параметры запуска
                self.opt.hyp = self.hyp
                # Инициализация объекта WandbLogger
                self.wandb = WandbLogger(self.opt, run_id)
                # Временное предупреждение, так как в версии Wandb выше 0.12.10 не поддерживаются вложенные артефакты
                # if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.11'):
                #    s = "YOLOv5 temporarily requires wandb version 0.12.10 or below. Some features may not work as expected."
                #    self.logger.warning(s)
            else:
                # Если условия не выполняются, устанавливаем атрибут wandb в None
                self.wandb = None

            # ClearML
            # Проверка, что библиотека clearml импортирована и 'clearml' включен в список поддерживаемых логгеров
            if clearml and 'clearml' in self.include:
                # Инициализация объекта ClearmlLogger
                self.clearml = ClearmlLogger(self.opt, self.hyp)
            else:
                # Если условия не выполняются, устанавливаем атрибут clearml в None
                self.clearml = None

            # Comet
            # Проверка, что библиотека comet_ml импортирована и 'comet' включен в список поддерживаемых логгеров
            if comet_ml and 'comet' in self.include:
                # Проверка, если параметр resume - это строка и начинается с 'comet://'
                if isinstance(self.opt.resume, str) and self.opt.resume.startswith("comet://"):
                    # Извлечение идентификатора запуска Comet из параметра resume
                    run_id = self.opt.resume.split("/")[-1]
                    # Инициализация объекта CometLogger с указанным идентификатором запуска
                    self.comet_logger = CometLogger(self.opt, self.hyp, run_id=run_id)
                else:
                    # Инициализация объекта CometLogger без указанного идентификатора запуска
                    self.comet_logger = CometLogger(self.opt, self.hyp)
            else:
                # Если условия не выполняются, устанавливаем атрибут comet_logger в None
                self.comet_logger = None

        @property
        def remote_dataset(self):
            # Получение словаря данных, если предоставлена ссылка на артефакт пользовательского набора данных
            data_dict = None
            # Если используется ClearML, получаем словарь данных от ClearML
            if self.clearml:
                data_dict = self.clearml.data_dict
            # Если используется Wandb, получаем словарь данных от Wandb
            if self.wandb:
                data_dict = self.wandb.data_dict
            # Если используется Comet, получаем словарь данных от Comet
            if self.comet_logger:
                data_dict = self.comet_logger.data_dict

            return data_dict

        def on_train_start(self):
            # Callback, выполняющийся в начале обучения
            if self.comet_logger:
                # Вызов метода on_train_start у объекта CometLogger
                self.comet_logger.on_train_start()

        def on_pretrain_routine_start(self):
            # Callback, выполняющийся в начале пред - тренировочного этапа
            if self.comet_logger:
                # Вызов метода on_pretrain_routine_start у объекта CometLogger
                self.comet_logger.on_pretrain_routine_start()

        def on_pretrain_routine_end(self, labels, names):
            # Callback, выполняющийся в конце пред - тренировочного этапа
            if self.plots:
                # Построение графиков меток
                plot_labels(labels, names, self.save_dir)
                # Получение списка файлов с метками обучения
                paths = self.save_dir.glob('*labels*.jpg')
                if self.wandb:
                    # Логирование изображений меток в Wandb
                    self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})
                # if self.clearml:
                #    pass  # ClearML сохраняет эти изображения автоматически с использованием хуков
                if self.comet_logger:
                    # Вызов метода on_pretrain_routine_end у объекта CometLogger
                    self.comet_logger.on_pretrain_routine_end(paths)

        def on_train_batch_end(self, model, ni, imgs, targets, paths, vals):
            # Создание словаря с логами на основе значений потерь
            log_dict = dict(zip(self.keys[0:3], vals))
            # Callback, выполняющийся в конце каждого обучающего батча
            # ni: количество интегрированных батчей (с момента начала обучения)
            if self.plots:
                if ni < 3:
                    # Имя файла для сохранения изображения батча
                    f = self.save_dir / f'train_batch{ni}.jpg'
                    # Построение изображений батча
                    plot_images(imgs, targets, paths, f)
                    if ni == 0 and self.tb and not self.opt.sync_bn:
                        # Логирование графа модели в TensorBoard
                        log_tensorboard_graph(self.tb, model, imgsz=(self.opt.imgsz, self.opt.imgsz))
                if ni == 10 and (self.wandb or self.clearml):
                    # Получение списка файлов с изображениями тренировки
                    files = sorted(self.save_dir.glob('train*.jpg'))
                    if self.wandb:
                        # Логирование изображений мозаик в Wandb
                        self.wandb.log({'Mosaics': [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})
                    if self.clearml:
                        # Логирование отладочных образцов в ClearML
                        self.clearml.log_debug_samples(files, title='Mosaics')

            if self.comet_logger:
                # Вызов метода on_train_batch_end у объекта CometLogger
                self.comet_logger.on_train_batch_end(log_dict, step=ni)

        def on_train_epoch_end(self, epoch):
            # Callback, выполняющийся в конце каждой эпохи обучения
            if self.wandb:
                # Установка текущей эпохи в объекте WandbLogger
                self.wandb.current_epoch = epoch + 1

            if self.comet_logger:
                # Вызов метода on_train_epoch_end у объекта CometLogger
                self.comet_logger.on_train_epoch_end(epoch)

        def on_val_start(self):
            # Callback, выполняющийся в начале валидации
            if self.comet_logger:
                # Вызов метода on_val_start у объекта CometLogger
                self.comet_logger.on_val_start()

        def on_val_image_end(self, pred, predn, path, names, im):
            # Callback, выполняющийся в конце обработки каждого изображения валидации
            if self.wandb:
                # Обработка одного изображения валидации в Wandb
                self.wandb.val_one_image(pred, predn, path, names, im)
            if self.clearml:
                # Логирование изображения с ограничивающими рамками в ClearML
                self.clearml.log_image_with_boxes(path, pred, names, im)

    def on_val_batch_end(self, batch_i, im, targets, paths, shapes, out):
        # Callback, выполняющийся в конце каждого валидационного батча
        if self.comet_logger:
            # Вызов метода on_val_batch_end у объекта CometLogger
            self.comet_logger.on_val_batch_end(batch_i, im, targets, paths, shapes, out)

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        # Callback, выполняющийся в конце процесса валидации
        if self.wandb or self.clearml:
            # Получение списка файлов с изображениями валидации
            files = sorted(self.save_dir.glob('val*.jpg'))
            if self.wandb:
                # Логирование изображений валидации в Wandb
                self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})
            if self.clearml:
                # Логирование отладочных образцов в ClearML для изображений валидации
                self.clearml.log_debug_samples(files, title='Validation')

        if self.comet_logger:
            # Вызов метода on_val_end у объекта CometLogger
            self.comet_logger.on_val_end(nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        # Callback, выполняющийся в конце каждой эпохи фитинга (обучение + валидация)
        x = dict(zip(self.keys, vals))
        if self.csv:
            # Путь к файлу results.csv для сохранения результатов
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # количество колонок
            # Создание заголовка файла, если он еще не существует
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')
            with open(file, 'a') as f:
                # Запись данных о текущей эпохе в файл results.csv
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            # Логирование значений в TensorBoard для каждой пары ключ - значение
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)
        elif self.clearml:  # если TensorBoard не используется, логировать в ClearML
            for k, v in x.items():
                title, series = k.split('/')
                # Логирование скалярного значения в ClearML
                self.clearml.task.get_logger().report_scalar(title, series, v, epoch)

        if self.wandb:
            if best_fitness == fi:
                # Сбор лучших результатов для текущей эпохи
                best_results = [epoch] + vals[3:7]
                for i, name in enumerate(self.best_keys):
                    # Логирование лучших результатов в-summary Wandb
                    self.wandb.wandb_run.summary[name] = best_results[i]
            # Логирование текущих результатов в Wandb
            self.wandb.log(x)
            # Завершение текущей эпохи в Wandb с пометкой о лучшем результате, если оно достигнуто
            self.wandb.end_epoch(best_result=best_fitness == fi)

        if self.clearml:
            # Сброс лимита изображений для текущей эпохи в ClearML
            self.clearml.current_epoch_logged_images = set()
            # Увеличение номера текущей эпохи в ClearML
            self.clearml.current_epoch += 1

        if self.comet_logger:
            # Вызов метода on_fit_epoch_end у объекта CometLogger
            self.comet_logger.on_fit_epoch_end(x, epoch=epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback, выполняющийся при сохранении модели
        if (epoch + 1) % self.opt.save_period == 0 and not final_epoch and self.opt.save_period != -1:
            if self.wandb:
                # Логирование сохраненной модели в Wandb
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)
            if self.clearml:
                # Обновление информации о выходной модели в ClearML
                self.clearml.task.update_output_model(model_path=str(last),
                                                      model_name='Latest Model',
                                                      auto_delete_file=False)

        if self.comet_logger:
            # Вызов метода on_model_save у объекта CometLogger
            self.comet_logger.on_model_save(last, epoch, final_epoch, best_fitness, fi)

    def on_train_end(self, last, best, epoch, results):
        # Callback, выполняющийся в конце обучения, т.е. при сохранении лучшей модели
        if self.plots:
            # Построение графиков результатов на основе файла results.csv
            plot_results(file=self.save_dir / 'results.csv')
        files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # фильтрация существующих файлов
        self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        if self.tb and not self.clearml:  # Эти изображения уже были сохранены ClearML, не нужно дублировать
            for f in files:
                # Добавление изображения в TensorBoard
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')

        if self.wandb:
            # Логирование результатов в Wandb
            self.wandb.log(dict(zip(self.keys[3:10], results)))
            # Логирование изображений результатов в Wandb
            self.wandb.log({"Results": [wandb.Image(str(f), caption=f.name) for f in files]})
            # Вызов wandb.log. TODO: Переработать это в WandbLogger.log_model
            if not self.opt.evolve:
                # Логирование артефакта модели в Wandb
                wandb.log_artifact(str(best if best.exists() else last),
                                   type='model',
                                   name=f'run_{self.wandb.wandb_run.id}_model',
                                   aliases=['latest', 'best', 'stripped'])
            # Завершение работы в Wandb
            self.wandb.finish_run()

        if self.clearml and not self.opt.evolve:
            # Обновление информации о выходной модели в ClearML как лучшей модели
            self.clearml.task.update_output_model(model_path=str(best if best.exists() else last),
                                                  name='Best Model',
                                                  auto_delete_file=False)

        if self.comet_logger:
            # Создание словаря с конечными результатами на основе определенных ключей и значений results
            final_results = dict(zip(self.keys[3:10], results))
            # Вызов метода on_train_end у объекта CometLogger с передачей файлов, директории сохранения, последней и лучшей модели, эпохи и конечных результатов
            self.comet_logger.on_train_end(files, self.save_dir, last, best, epoch, final_results)

        def on_params_update(self, params: dict):
            # Обновление гиперпараметров или настроек эксперимента
            if self.wandb:
                # Обновление настроек запуска Wandb с новыми параметрами, разрешая изменение значений
                self.wandb.wandb_run.config.update(params, allow_val_change=True)
            if self.comet_logger:
                # Вызов метода on_params_update у объекта CometLogger для обновления параметров
                self.comet_logger.on_params_update(params)

        class GenericLogger:
            """
            YOLOv5 Общий логгер для ведения логов не относящихся к конкретной задаче
            Использование: from utils.loggers import GenericLogger; logger = GenericLogger(...)
            Аргументы
                opt:             Аргументы запуска
                console_logger:  Консольный логгер
                include:         Логгеры для включения
            """

            def __init__(self, opt, console_logger, include=('tb', 'wandb')):
                # Инициализация стандартных логгеров
                self.save_dir = Path(opt.save_dir)
                self.include = include
                self.console_logger = console_logger
                self.csv = self.save_dir / 'results.csv'  # CSV логгер
                if 'tb' in self.include:
                    # Префикс для сообщения о TensorBoard
                    prefix = colorstr('TensorBoard: ')
                    # Вывод сообщения в консольный логгер о том, как запустить TensorBoard и открыть его в браузере
                    self.console_logger.info(
                        f"{prefix}Start with 'tensorboard --logdir {self.save_dir.parent}', view at http://localhost:6006/")
                    # Инициализация объекта SummaryWriter для TensorBoard с указанной директорией сохранения
                    self.tb = SummaryWriter(str(self.save_dir))

                if wandb and 'wandb' in self.include:
                    # Инициализация работы Wandb с указанием проекта, имени (если не "exp") и конфигурации
                    self.wandb = wandb.init(project=web_project_name(str(opt.project)),
                                            name=None if opt.name == "exp" else opt.name,
                                            config=opt)
                else:
                    # Если Wandb не включен или не импортирован, устанавливаем атрибут wandb в None
                    self.wandb = None

            def log_metrics(self, metrics, epoch):
                # Логирование словаря метрик во все логгеры
                if self.csv:
                    # Извлечение ключей и значений из словаря метрик
                    keys, vals = list(metrics.keys()), list(metrics.values())
                    n = len(metrics) + 1  # количество колонок
                    # Создание заголовка файла CSV, если он еще не существует
                    s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')
                    with open(self.csv, 'a') as f:
                        # Запись данных о эпохе и метриках в файл CSV
                        f.write(s + ('%23.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

                if self.tb:
                    # Логирование каждого метрики в TensorBoard
                    for k, v in metrics.items():
                        self.tb.add_scalar(k, v, epoch)

                if self.wandb:
                    # Логирование метрик в Wandb с указанием шага (эпохи)
                    self.wandb.log(metrics, step=epoch)

            def log_images(self, files, name='Images', epoch=0):
                # Логирование изображений во все логгеры
                files = [Path(f) for f in
                         (files if isinstance(files, (tuple, list)) else [files])]  # преобразование в Path
                files = [f for f in files if f.exists()]  # фильтрация существующих файлов

                if self.tb:
                    # Добавление каждого изображения в TensorBoard
                    for f in files:
                        self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')

                if self.wandb:
                    # Логирование изображений в Wandb с указанием шага (эпохи) и имени набора изображений
                    self.wandb.log({name: [wandb.Image(str(f), caption=f.name) for f in files]}, step=epoch)

            def log_graph(self, model, imgsz=(640, 640)):
                # Логирование графа модели во все логгеры
                if self.tb:
                    # Вызов функции для логирования графа модели в TensorBoard
                    log_tensorboard_graph(self.tb, model, imgsz)

            def log_model(self, model_path, epoch=0, metadata={}):
                # Логирование модели во все логгеры
                if self.wandb:
                    # Создание артефакта Wandb для модели с указанием имени, типа и метаданных
                    art = wandb.Artifact(name=f"run_{wandb.run.id}_model", type="model", metadata=metadata)
                    # Добавление файла модели в артефакт
                    art.add_file(str(model_path))
                    # Логирование артефакта модели в Wandb
                    wandb.log_artifact(art)

            def update_params(self, params):
                # Обновление параметров, которые были залогированы
                if self.wandb:
                    # Обновление настроек запуска Wandb с новыми параметрами, разрешая изменение значений
                    wandb.run.config.update(params, allow_val_change=True)

        def log_tensorboard_graph(tb, model, imgsz=(640, 640)):
            # Логирование графа модели в TensorBoard
            try:
                # Получение параметра модели для определения устройства и типа
                p = next(model.parameters())
                imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz  # расширение размерности
                # Создание входного изображения (должно быть заполнено нулями)
                im = torch.zeros((1, 3, *imgsz)).to(p.device).type_as(p)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')  # подавление предупреждения при трассировке JIT
                    # Добавление графа модели в TensorBoard с использованием трассировки JIT
                    tb.add_graph(torch.jit.trace(de_parallel(model), im, strict=False), [])
            except Exception as e:
                # Вывод предупреждения в LOGGER в случае ошибки при визуализации графа в TensorBoard
                LOGGER.warning(f'WARNING ⚠️ TensorBoard graph visualization failure {e}')

        def web_project_name(project):
            # Преобразование локального имени проекта в имя проекта для веб - интерфейса
            if not project.startswith('runs/train'):
                return project
            suffix = '-Classify' if project.endswith('-cls') else '-Segment' if project.endswith('-seg') else ''
            return f'YOLOv5{suffix}'