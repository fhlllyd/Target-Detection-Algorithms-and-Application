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

# Получаем абсолютный путь к текущему файлу
FILE = Path(__file__).resolve()
# Определяем корневую директорию YOLOv5
ROOT = FILE.parents[1]
# Добавляем корневую директорию в системный путь, если ее там еще нет
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# Преобразуем корневую директорию в относительный путь
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

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
    weights=ROOT / 'yolov5s-seg.pt',  # Путь к файлу с весами модели
    source=ROOT / 'data/images',  # Источник данных (файл, директория, URL и т.д.)
    data=ROOT / 'data/coco128.yaml',  # Путь к файлу с описанием набора данных
    imgsz=(640, 640),  # Размер изображения для инференса (высота, ширина)
    conf_thres=0.25,  # Порог уверенности для фильтрации предсказаний
    iou_thres=0.45,  # Порог IoU для не максимального подавления (NMS)
    max_det=1000,  # Максимальное количество детекций на изображение
    device='',  # Устройство для выполнения (GPU или CPU)
    view_img=False,  # Показывать результаты инференса
    save_txt=False,  # Сохранять результаты в текстовых файлах
    save_conf=False,  # Сохранять уверенности в текстовых файлах с результатами
    save_crop=False,  # Сохранять вырезанные области с предсказанными объектами
    nosave=False,  # Не сохранять изображения или видео с результатами
    classes=None,  # Фильтровать предсказания по классам
    agnostic_nms=False,  # Использовать класс-агностичное не максимальное подавление
    augment=False,  # Использовать аугментацию при инференсе
    visualize=False,  # Визуализировать признаки
    update=False,  # Обновить все модели
    project=ROOT / 'runs/predict-seg',  # Директория для сохранения результатов
    name='exp',  # Имя поддиректории для сохранения результатов
    exist_ok=False,  # Разрешить использование существующей директории для сохранения
    line_thickness=3,  # Толщина рамок вокруг предсказанных объектов
    hide_labels=False,  # Скрыть метки классов на изображениях
    hide_conf=False,  # Скрыть уверенности на изображениях
    half=False,  # Использовать половинную точность (FP16) при инференсе
    dnn=False,  # Использовать OpenCV DNN для инференса с ONNX моделью
    vid_stride=1,  # Шаг при обработке видеокадров
    retina_masks=False,
):
    # Преобразуем источник данных в строку
    source = str(source)
    # Определяем, нужно ли сохранять изображения с результатами инференса
    save_img = not nosave and not source.endswith('.txt')
    # Проверяем, является ли источник файлом с поддерживаемым форматом изображения или видео
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # Проверяем, является ли источник URL
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Проверяем, является ли источник веб-камерой, текстовым файлом или URL
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # Проверяем, является ли источник скриншотом
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        # Если источник - URL и файл, скачиваем файл
        source = check_file(source)

    # Создаем директорию для сохранения результатов
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    # Создаем поддиректорию для сохранения текстовых файлов с результатами, если нужно
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Загружаем модель
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # Получаем шаг модели, имена классов и флаг, указывающий на использование PyTorch
    stride, names, pt = model.stride, model.names, model.pt
    # Проверяем размер изображения на соответствие шагу модели
    imgsz = check_img_size(imgsz, s=stride)

    # Создаем загрузчик данных
    bs = 1  # Размер батча
    if webcam:
        # Проверяем, можно ли показывать изображения
        view_img = check_imshow(warn=True)
        # Создаем загрузчик для видеопотока
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        # Создаем загрузчик для скриншотов
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        # Создаем загрузчик для изображений
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # Инициализируем переменные для записи видео
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Производим прогрев модели
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    # Счетчик обработанных изображений
    seen = 0
    # Список окон для отображения изображений
    windows = []
    # Счетчики времени для разных этапов обработки
    dt = (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # Преобразуем изображение в тензор и перемещаем его на устройство
            im = torch.from_numpy(im).to(model.device)
            # Преобразуем тип данных изображения в полную или половинную точность
            im = im.half() if model.fp16 else im.float()
            # Нормализуем изображение
            im /= 255
            if len(im.shape) == 3:
                # Добавляем размерность батча, если ее нет
                im = im[None]

        # Выполняем инференс
        with dt[1]:
            if visualize:
                # Создаем директорию для визуализации признаков
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True)
            # Получаем предсказания и прототипы масок
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

        # Применяем не максимальное подавление (NMS)
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Второй этап классификации (необязательно)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Обработка предсказаний
            for i, det in enumerate(pred):  # для каждого изображения
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # преобразование в Path
                save_path = str(save_dir / p.name)  # путь для сохранения изображения
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # путь для .txt
                s += '%gx%g ' % im.shape[2:]  # строка для вывода
                imc = im0.copy() if save_crop else im0  # копия для вырезания
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Обработка масок
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    # Масштаббоксов к исходному изображению
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Сегменты
                    if save_txt:
                        segments = reversed(masks2segments(masks))
                        segments = [scale_segments(im.shape[2:], x, im0.shape, normalize=True) for x in segments]

                    # Вывод результатов
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # количество детекций
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # добавление в строку

                    # Рисование масок
                    annotator.masks(masks,
                                    colors=[colors(x, True) for x in det[:, 5]],
                                    im_gpu=None if retina_masks else im[i])

                    # Запись результатов
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        if save_txt:  # Запись в файл
                            segj = segments[j].reshape(-1)  # преобразование сегмента
                            line = (cls, *segj, conf) if save_conf else (cls, *segj)  # формат метки
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Добавление рамки
                            c = int(cls)  # класс как целое число
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            # Сохранение выреза
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Отображение результатов
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        # Создание окна с возможностью изменения размера
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # выход по нажатию Q
                        exit()

                # Сохранение результатов
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # видео или поток
                        if vid_path[i] != save_path:  # новый видеофайл
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # закрытие предыдущего записчика
                            if vid_cap:  # видео
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # поток
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # формат mp4
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Вывод времени инференса
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Итоговая информация
        t = tuple(x.t / seen * 1E3 for x in dt)  # время в миллисекундах
        LOGGER.info(f'Скорость: %.1fms предобработка, %.1fms инференс, %.1fms NMS на изображение' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} меток сохранено в {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Результаты сохранены в {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # обновление модели


def parse_opt():
    # Создание парсера аргументов командной строки
    parser = argparse.ArgumentParser()

    # Параметры для загрузки модели
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'runs/train-seg/wheel/weights/best.pt',
                        help='путь к файлу модели')
    parser.add_argument('--source', type=str,
                        default=ROOT / 'C:/Users/C110/Desktop/积水检测实验照片/wheel-2.jpg',
                        help='источник данных (файл/директория/URL/экран/0 для веб-камеры)')
    parser.add_argument('--data', type=str,
                        default=ROOT / 'data/coco128.yaml',
                        help='путь к файлу описания датасета (опционально)')

    # Параметры инференса
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int,
                        default=[640],
                        help='размер изображения для инференса (высота, ширина)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='порог уверенности для фильтрации предсказаний')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='порог IoU для не максимального подавления (NMS)')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='максимальное количество детекций на изображение')

    # Параметры устройства
    parser.add_argument('--device', default='',
                        help='устройство для вычислений (например, 0, 0,1,2,3 или cpu)')

    # Параметры визуализации
    parser.add_argument('--view-img', action='store_true',
                        help='показывать результаты инференса на экране')
    parser.add_argument('--save-txt', action='store_true',
                        help='сохранять результаты в текстовых файлах')
    parser.add_argument('--save-conf', action='store_true',
                        help='сохранять значения уверенности в текстовых файлах')
    parser.add_argument('--save-crop', action='store_true',
                        help='сохранять вырезанные области с детекциями')
    parser.add_argument('--nosave', action='store_true',
                        help='не сохранять изображения/видео с результатами')

    # Параметры фильтрации
    parser.add_argument('--classes', nargs='+', type=int,
                        help='фильтрация по классам (например, --classes 0 или 0 2 3)')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='использовать класс-агностичное не максимальное подавление')
    parser.add_argument('--augment', action='store_true',
                        help='использовать аугментацию при инференсе')

    # Параметры сохранения
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg',
                        help='директория для сохранения результатов')
    parser.add_argument('--name', default='exp',
                        help='имя поддиректории для сохранения результатов')
    parser.add_argument('--exist-ok', action='store_true',
                        help='разрешить использование существующей директории')

    # Прочие параметры
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='толщина рамок вокруг детекций')
    parser.add_argument('--hide-labels', default=False, action='store_true',
                        help='скрывать метки классов на изображениях')
    parser.add_argument('--hide-conf', default=False, action='store_true',
                        help='скрывать значения уверенности на изображениях')
    parser.add_argument('--half', action='store_true',
                        help='использовать половинную точность (FP16)')
    parser.add_argument('--dnn', action='store_true',
                        help='использовать OpenCV DNN для инференса с ONNX моделью')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='шаг при обработке видеокадров')
    parser.add_argument('--retina-masks', action='store_true',
                        help='отображать маски в нативном разрешении')

    # Парсинг аргументов
    opt = parser.parse_args()
    # Корректировка размера изображения
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    # Вывод аргументов
    print_args(vars(opt))
    return opt


def main(opt):
    # Проверка наличия необходимых библиотек
    check_requirements(exclude=('tensorboard', 'thop'))
    # Запуск основной функции инференса
    run(**vars(opt))


if __name__ == "__main__":
    # Парсинг аргументов и запуск программы
    opt = parse_opt()
    main(opt)