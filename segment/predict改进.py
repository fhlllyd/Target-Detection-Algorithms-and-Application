# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Корневая директория YOLOv5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Добавляем в системный путь
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Относительный путь

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-seg.pt',  # Путь к модели (можно указать несколько)
        source=ROOT / 'data/images',  # Источник данных (файл/директория/URL/экран/0 для веб-камеры)
        data=ROOT / 'data/coco128.yaml',  # Конфигурационный файл датасета
        imgsz=(640, 640),  # Размер изображения для инференса (высота, ширина)
        conf_thres=0.25,  # Порог уверенности для фильтрации объектов
        iou_thres=0.45,  # Порог IoU для NMS
        max_det=1000,  # Максимальное количество детекций на изображение
        device='',  # Устройство (GPU: 0, 0,1,2 или CPU)
        view_img=False,  # Показывать результаты на экране
        save_txt=False,  # Сохранять результаты в текстовые файлы
        save_conf=False,  # Сохранять значения уверенности в текстовых файлах
        save_crop=False,  # Сохранять вырезанные объекты
        nosave=False,  # Не сохранять изображения/видео
        classes=None,  # Фильтр классов (например: --classes 0 2 3)
        agnostic_nms=False,  # Класс-агностичное NMS
        augment=False,  # Использовать аугментацию при инференсе
        visualize=False,  # Визуализировать признаки
        update=False,  # Обновить веса модели
        project=ROOT / 'runs/predict-seg',  # Директория для сохранения результатов
        name='exp',  # Имя эксперимента
        exist_ok=False,  # Разрешить существующую директорию
        line_thickness=3,  # Толщина рамок
        hide_labels=False,  # Скрыть метки классов
        hide_conf=False,  # Скрыть значения уверенности
        half=False,  # Использовать FP16
        dnn=False,  # Использовать OpenCV DNN для ONNX
        vid_stride=1,  # Шаг обработки видеокадров
        retina_masks=False,  # Рисовать маски в нативном разрешении
):
    source = str(source)
    # Определяем, нужно ли сохранять изображения
    save_img = not nosave and not source.endswith('.txt')
    # Проверяем тип источника данных
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')

    if is_url and is_file:
        source = check_file(source)  # Скачиваем файл по URL

    # Создаем директорию для результатов
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Загружаем модель
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # Проверяем размер изображения

    # Создаем загрузчик данных
    bs = 1  # Размер батча
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Прогрев модели
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # Преобразуем изображение в тензор
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # Преобразование в FP16/FP32
            im /= 255  # Нормализация
            if len(im.shape) == 3:
                im = im[None]  # Добавляем размерность батча

        # Инференс
        with dt[1]:
            visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize_path)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Дополнительная классификация (если требуется)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[1]  # Корневая директория YOLOv5
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # Добавляем в системный путь
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Относительный путь

        from models.common import DetectMultiBackend
        from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
        from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
                                   colorstr, cv2,
                                   increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                                   strip_optimizer, xyxy2xywh)
        from utils.plots import Annotator, colors, save_one_box
        from utils.segment.general import masks2segments, process_mask
        from utils.torch_utils import select_device, smart_inference_mode

        @smart_inference_mode()
        def run(
                weights=ROOT / 'yolov5s-seg.pt',  # Путь к модели (можно указать несколько)
                source=ROOT / 'data/images',  # Источник данных (файл/директория/URL/экран/0 для веб-камеры)
                data=ROOT / 'data/coco128.yaml',  # Конфигурационный файл датасета
                imgsz=(640, 640),  # Размер изображения для инференса (высота, ширина)
                conf_thres=0.25,  # Порог уверенности для фильтрации объектов
                iou_thres=0.45,  # Порог IoU для NMS
                max_det=1000,  # Максимальное количество детекций на изображение
                device='',  # Устройство (GPU: 0, 0,1,2 или CPU)
                view_img=False,  # Показывать результаты на экране
                save_txt=False,  # Сохранять результаты в текстовые файлы
                save_conf=False,  # Сохранять значения уверенности в текстовых файлах
                save_crop=False,  # Сохранять вырезанные объекты
                nosave=False,  # Не сохранять изображения/видео
                classes=None,  # Фильтр классов (например: --classes 0 2 3)
                agnostic_nms=False,  # Класс-агностичное NMS
                augment=False,  # Использовать аугментацию при инференсе
                visualize=False,  # Визуализировать признаки
                update=False,  # Обновить веса модели
                project=ROOT / 'runs/predict-seg',  # Директория для сохранения результатов
                name='exp',  # Имя эксперимента
                exist_ok=False,  # Разрешить существующую директорию
                line_thickness=3,  # Толщина рамок
                hide_labels=False,  # Скрыть метки классов
                hide_conf=False,  # Скрыть значения уверенности
                half=False,  # Использовать FP16
                dnn=False,  # Использовать OpenCV DNN для ONNX
                vid_stride=1,  # Шаг обработки видеокадров
                retina_masks=False,  # Рисовать маски в нативном разрешении
        ):
            source = str(source)
            # Определяем, нужно ли сохранять изображения
            save_img = not nosave and not source.endswith('.txt')
            # Проверяем тип источника данных
            is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
            is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
            webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
            screenshot = source.lower().startswith('screen')

            if is_url and is_file:
                source = check_file(source)  # Скачиваем файл по URL

            # Создаем директорию для результатов
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

            # Загружаем модель
            device = select_device(device)
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(imgsz, s=stride)  # Проверяем размер изображения

            # Создаем загрузчик данных
            bs = 1  # Размер батча
            if webcam:
                view_img = check_imshow(warn=True)
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                bs = len(dataset)
            elif screenshot:
                dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            vid_path, vid_writer = [None] * bs, [None] * bs

            # Прогрев модели
            model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))

            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                    # Преобразуем изображение в тензор
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # Преобразование в FP16/FP32
                    im /= 255  # Нормализация
                    if len(im.shape) == 3:
                        im = im[None]  # Добавляем размерность батча

                # Инференс
                with dt[1]:
                    visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred, proto = model(im, augment=augment, visualize=visualize_path)[:2]

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det,
                                               nm=32)

                # Дополнительная классификация (если требуется)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

def parse_opt():
    parser = argparse.ArgumentParser()

    # Параметры модели
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'runs/train-seg/exp6/weights/best.pt',
                        help='путь к файлам с весами модели')
    parser.add_argument('--source', type=str,
                        default=ROOT / 'C:/Users/C110/Desktop/car3.jpg',
                        help='источник данных (файл/директория/URL/экран/0 для веб-камеры)')
    parser.add_argument('--data', type=str,
                        default=ROOT / 'data/coco128.yaml',
                        help='путь к конфигурационному файлу датасета (опционально)')

    # Параметры инференса
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int,
                        default=[640],
                        help='размер изображения для инференса (высота, ширина)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='порог уверенности для фильтрации объектов')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='порог IoU для не максимального подавления (NMS)')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='максимальное количество детекций на изображение')

    # Параметры устройства
    parser.add_argument('--device', default='',
                        help='устройство для вычислений (например, 0, 0,1,2,3 или cpu)')

    # Параметры визуализации и сохранения
    parser.add_argument('--view-img', action='store_true',
                        help='показывать результаты на экране')
    parser.add_argument('--save-txt', action='store_true',
                        help='сохранять результаты в текстовые файлы')
    parser.add_argument('--save-conf', action='store_true',
                        help='сохранять значения уверенности в текстовых файлах')
    parser.add_argument('--save-crop', action='store_true',
                        help='сохранять вырезанные объекты')
    parser.add_argument('--nosave', action='store_true',
                        help='не сохранять изображения/видео')

    # Параметры фильтрации
    parser.add_argument('--classes', nargs='+', type=int,
                        help='фильтр классов (например, --classes 0 2 3)')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='класс-агностичное не максимальное подавление')
    parser.add_argument('--augment', action='store_true',
                        help='использовать аугментацию при инференсе')

    # Параметры сохранения результатов
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg',
                        help='директория для сохранения результатов')
    parser.add_argument('--name', default='exp',
                        help='имя эксперимента')
    parser.add_argument('--exist-ok', action='store_true',
                        help='разрешить существующую директорию')

    # Прочие параметры
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='толщина рамок вокруг детекций')
    parser.add_argument('--hide-labels', default=False, action='store_true',
                        help='скрыть метки классов')
    parser.add_argument('--hide-conf', default=False, action='store_true',
                        help='скрыть значения уверенности')
    parser.add_argument('--half', action='store_true',
                        help='использовать половинную точность (FP16)')
    parser.add_argument('--dnn', action='store_true',
                        help='использовать OpenCV DNN для ONNX-инференса')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='шаг обработки видеокадров')
    parser.add_argument('--retina-masks', action='store_true',
                        help='отображать маски в нативном разрешении')

    opt = parser.parse_args()
    # Корректировка размера изображения
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    # Проверка наличия необходимых библиотек
    check_requirements(exclude=('tensorboard', 'thop'))
    # Запуск основной функции инференса
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)