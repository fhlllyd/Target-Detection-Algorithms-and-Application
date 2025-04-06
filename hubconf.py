# YOLOv5 🚀 от Ultralytics, лицензия GPL-3.0
"""
Модели PyTorch Hub https://pytorch.org/hub/ultralytics_yolov5

Использование:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # официальная модель
    model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')  # из ветки
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')  # пользовательская/локальная модель
    model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')  # локальный репозиторий
"""

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """Создает или загружает модель YOLOv5

    Аргументы:
        name (str): название модели 'yolov5s' или путь 'path/to/best.pt'
        pretrained (bool): загружать предобученные веса в модель
        channels (int): количество входных каналов
        classes (int): количество классов модели
        autoshape (bool): применить обертку YOLOv5 .autoshape() к модели
        verbose (bool): выводить всю информацию на экран
        device (str, torch.device, None): устройство для параметров модели

    Возвращает:
        Модель YOLOv5
    """
    from pathlib import Path

    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
    from utils.downloads import attempt_download
    from utils.general import LOGGER, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device

    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(exclude=('opencv-python', 'tensorboard', 'thop'))
    name = Path(name)
    path = name.with_suffix('.pt') if name.suffix == '' and not name.is_dir() else name  # путь к контрольной точке
    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)  # детекционная модель
                if autoshape:
                    if model.pt and isinstance(model.model, ClassificationModel):
                        LOGGER.warning('WARNING ⚠️ YOLOv5 ClassificationModel еще не совместим с AutoShape. '
                                       'Вы должны передавать torch тензоры в формате BCHW в эту модель, например, shape(1,3,224,224).')
                    elif model.pt and isinstance(model.model, SegmentationModel):
                        LOGGER.warning('WARNING ⚠️ YOLOv5 SegmentationModel еще не совместим с AutoShape. '
                                       'Вы не сможете выполнить инференс с этой моделью.')
                    else:
                        model = AutoShape(model)  # для файлов/URI/PIL/cv2/np ввода и NMS
            except Exception:
                model = attempt_load(path, device=device, fuse=False)  # произвольная модель
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{path.stem}.yaml'))[0]  # путь к model.yaml
            model = DetectionModel(cfg, channels, classes)  # создание модели
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # загрузка
                csd = ckpt['model'].float().state_dict()  # состояние контрольной точки в виде FP32
                csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # пересечение
                model.load_state_dict(csd, strict=False)  # загрузка
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # установка атрибута имен классов
        if not verbose:
            LOGGER.setLevel(logging.INFO)  # сброс до значения по умолчанию
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = f'{e}. Кэш может быть устаревшим, попробуйте `force_reload=True` или обратитесь за помощью на {help_url}.'
        raise Exception(s) from e


def custom(path='path/to/model.pt', autoshape=True, _verbose=True, device=None):
    # Пользовательская или локальная модель YOLOv5
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # Модель YOLOv5-nano https://github.com/ultralytics/yolov5
    return _create('yolov5n', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # Модель YOLOv5-small https://github.com/ultralytics/yolov5
    return _create('yolov5s', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # Модель YOLOv5-medium https://github.com/ultralytics/yolov5
    return _create('yolov5m', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # Модель YOLOv5-large https://github.com/ultralytics/yolov5
    return _create('yolov5l', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # Модель YOLOv5-xlarge https://github.com/ultralytics/yolov5
    return _create('yolov5x', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # Модель YOLOv5-nano-P6 https://github.com/ultralytics/yolov5
    return _create('yolov5n6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # Модель YOLOv5-small-P6 https://github.com/ultralytics/yolov5
    return _create('yolov5s6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # Модель YOLOv5-medium-P6 https://github.com/ultralytics/yolov5
    return _create('yolov5m6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # Модель YOLOv5-large-P6 https://github.com/ultralytics/yolov5
    return _create('yolov5l6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # Модель YOLOv5-xlarge-P6 https://github.com/ultralytics/yolov5
    return _create('yolov5x6', pretrained, channels, classes, autoshape, _verbose, device)


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2, print_args

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s', help='название модели')
    opt = parser.parse_args()
    print_args(vars(opt))

    # Модель
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    # model = custom(path='path/to/model.pt')  # пользовательская

    # Изображения
    imgs = [
        'data/images/zidane.jpg',  # имя файла
        Path('data/images/zidane.jpg'),  # Path
        'https://ultralytics.com/images/zidane.jpg',  # URI
        cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
        Image.open('data/images/bus.jpg'),  # PIL
        np.zeros((320, 640, 3))]  # numpy

    # Инференс
    results = model(imgs, size=320)  # пакетный инференс

    # Результаты
    results.print()
    results.save()