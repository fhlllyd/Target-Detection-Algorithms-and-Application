# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 classification inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python classify/predict.py --weights yolov5s-cls.pt --source 0                               # webcam
                                                                   img.jpg                         # image
                                                                   vid.mp4                         # video
                                                                   screen                          # screenshot
                                                                   path/                           # directory
                                                                   'path/*.jpg'                    # glob
                                                                   'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                   'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python classify/predict.py --weights yolov5s-cls.pt                 # PyTorch
                                           yolov5s-cls.torchscript        # TorchScript
                                           yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                           yolov5s-cls_openvino_model     # OpenVINO
                                           yolov5s-cls.engine             # TensorRT
                                           yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                           yolov5s-cls_saved_model        # TensorFlow SavedModel
                                           yolov5s-cls.pb                 # TensorFlow GraphDef
                                           yolov5s-cls.tflite             # TensorFlow Lite
                                           yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                           yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Получить абсолютный путь текущего файла
FILE = Path(__file__).resolve()
# Получить корневую директорию YOLOv5
ROOT = FILE.parents[1]
# Добавить корневую директорию в системный путь, если ее там нет
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# Получить относительный путь к корневой директории
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Импорт класса DetectMultiBackend из модуля models.common
from models.common import DetectMultiBackend
# Импорт функции classify_transforms из модуля utils.augmentations
from utils.augmentations import classify_transforms
# Импорт необходимых переменных и классов из модуля utils.dataloaders
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# Импорт различных функций и переменных из модуля utils.general
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, print_args, strip_optimizer)
# Импорт класса Annotator из модуля utils.plots
from utils.plots import Annotator
# Импорт функций из модуля utils.torch_utils
from utils.torch_utils import select_device, smart_inference_mode

# Декоратор для интеллектуального режима вывода
@smart_inference_mode()
def run(
        # Путь к файлу с весами модели
        weights=ROOT / 'yolov5s-cls.pt',
        # Источник данных (файл, директория, URL и т.д.)
        source=ROOT / 'data/images',
        # Путь к файлу с данными датасета
        data=ROOT / 'data/coco128.yaml',
        # Размер изображения для вывода (высота, ширина)
        imgsz=(224, 224),
        # Устройство CUDA или CPU для вывода
        device='',
        # Показать результаты вывода
        view_img=False,
        # Сохранить результаты в текстовый файл
        save_txt=False,
        # Не сохранять изображения/видео
        nosave=False,
        # Использовать расширенный вывод
        augment=False,
        # Визуализировать признаки
        visualize=False,
        # Обновить все модели
        update=False,
        # Директория для сохранения результатов
        project=ROOT / 'runs/predict-cls',
        # Имя для сохранения результатов
        name='exp',
        # Разрешить использование существующей директории
        exist_ok=False,
        # Использовать половинную точность (FP16) для вывода
        half=False,
        # Использовать OpenCV DNN для вывода ONNX моделей
        dnn=False,
        # Шаг для обработки видеокадров
        vid_stride=1,
):
    # Преобразовать источник в строку
    source = str(source)
    # Флаг для сохранения изображений
    save_img = not nosave and not source.endswith('.txt')
    # Проверка, является ли источник файлом изображения или видео
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # Проверка, является ли источник URL
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Проверка, является ли источник веб-камерой, текстовым файлом или URL
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # Проверка, является ли источник скриншотом
    screenshot = source.lower().startswith('screen')
    # Если источник - URL и файл, то проверить и загрузить файл
    if is_url and is_file:
        source = check_file(source)

    # Создать директорию для сохранения результатов
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    # Создать поддиректорию для меток, если нужно сохранять текстовые файлы
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Выбрать устройство для вывода
    device = select_device(device)
    # Загрузить модель
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # Получить шаг, имена классов и флаг PyTorch модели
    stride, names, pt = model.stride, model.names, model.pt
    # Проверить размер изображения
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    # Размер пакета
    bs = 1
    # Если источник - веб-камера
    if webcam:
        # Проверить, можно ли показать изображение
        view_img = check_imshow(warn=True)
        # Создать объект загрузки видеопотоков
        dataset = LoadStreams(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
        # Установить размер пакета равным количеству видеопотоков
        bs = len(dataset)
    # Если источник - скриншот
    elif screenshot:
        # Создать объект загрузки скриншотов
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    # В противном случае
    else:
        # Создать объект загрузки изображений
        dataset = LoadImages(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
    # Инициализировать переменные для путей к видео и видео-писателей
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # Произвести прогрев модели
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    # Количество обработанных изображений
    seen = 0
    # Список окон для отображения изображений
    windows = []
    # Объекты для измерения времени
    dt = (Profile(), Profile(), Profile())
    # Итерировать по изображениям в наборе данных
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # Преобразовать изображение в тензор и перенести на устройство модели
            im = torch.Tensor(im).to(model.device)
            # Преобразовать тип данных изображения
            im = im.half() if model.fp16 else im.float()
            # Добавить размерность пакета, если ее нет
            if len(im.shape) == 3:
                im = im[None]

                # Inference
        with dt[1]:
            # Получить результаты модели
            results = model(im)

        # Post-process
        with dt[2]:
            # Применить функцию softmax для получения вероятностей
            pred = F.softmax(results, dim=1)

            # Process predictions
        # Итерировать по предсказаниям для каждого изображения
        for i, prob in enumerate(pred):
            # Увеличить счетчик обработанных изображений
            seen += 1
            # Если источник - веб-камера
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            # В противном случае
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # Преобразовать путь в объект Path
            p = Path(p)
            # Получить путь для сохранения изображения
            save_path = str(save_dir / p.name)
            # Получить путь для сохранения текстового файла с результатами
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')

            # Добавить размер изображения в строку вывода
            s += '%gx%g ' % im.shape[2:]
            # Создать объект для аннотации изображения
            annotator = Annotator(im0, example=str(names), pil=True)

            # Print results
            # Получить индексы топ-5 классов с наибольшими вероятностями
            top5i = prob.argsort(0, descending=True)[:5].tolist()
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "

            # Write results
            # Создать строку с текстом для записи в файл или отображения на изображении
            text = '\n'.join(f'{prob[j]:.2f} {names[j]}' for j in top5i)
            # Если нужно сохранить изображение или показать его
            if save_img or view_img:
                # Добавить текст на изображение
                annotator.text((32, 32), text, txt_color=(255, 255, 255))
            # Если нужно сохранить результаты в текстовый файл
            if save_txt:
                # Открыть файл и записать текст
                with open(f'{txt_path}.txt', 'a') as f:
                    f.write(text + '\n')

            # Stream results
            # Получить изображение с аннотациями
            im0 = annotator.result()
            # Если нужно показать изображение
            if view_img:
                # Если система Linux и окно еще не создано
                if platform.system() == 'Linux' and p not in windows:
                    # Добавить окно в список
                    windows.append(p)
                    # Создать окно с возможностью изменения размера
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    # Установить размер окна
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # Показать изображение в окне
                cv2.imshow(str(p), im0)
                # Подождать 1 миллисекунду
                cv2.waitKey(1)

                # Save results (image with detections)
            # Если нужно сохранить изображение
            if save_img:
                # Если источник - изображение
                if dataset.mode == 'image':
                    # Сохранить изображение
                    cv2.imwrite(save_path, im0)
                # В противном случае (видео или поток)
                else:
                    # Если путь к видео изменился
                    if vid_path[i] != save_path:
                        # Обновить путь к видео
                        vid_path[i] = save_path
                        # Если видео писатель уже создан, закрыть его
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                            # Если источник - видео
                        if vid_cap:
                            # Получить частоту кадров
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            # Получить ширину кадра
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            # Получить высоту кадра
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        # В противном случае (поток)
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # Принудительно установить расширение файла .mp4
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        # Создать новый видео писатель
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # Записать кадр в видео
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # Вывести время вывода модели
        LOGGER.info(f"{s}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    # Вычислить время обработки на одно изображение
    t = tuple(x.t / seen * 1E3 for x in dt)
    # Вывести информацию о времени обработки
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # Если нужно сохранить текстовые файлы или изображения
    if save_txt or save_img:
        # Создать строку с информацией о сохраненных метках
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # Вывести информацию о сохранении результатов
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # Если нужно обновить модель
    if update:
        # Обновить модель
        strip_optimizer(weights[0])


def parse_opt():
    # Создать объект парсера аргументов командной строки
    parser = argparse.ArgumentParser()
    # Добавить аргумент для пути к файлу с весами модели
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-cls.pt', help='model path(s)')
    # Добавить аргумент для источника данных
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    # Добавить аргумент для пути к файлу с данными датасета
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    # Добавить аргумент для размера изображения для вывода
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[224], help='inference size h,w')
    # Добавить аргумент для устройства CUDA или CPU
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # Добавить аргумент для показа результатов
    parser.add_argument('--view-img', action='store_true', help='show results')
    # Добавить аргумент для сохранения результатов в текстовый файл
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # Добавить аргумент для не сохранения изображений/видео
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # Добавить аргумент для использования расширенного вывода
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # Добавить аргумент для визуализации признаков
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    # Добавить аргумент для обновления всех моделей
    parser.add_argument('--update', action='store_true', help='update all models')
    # Добавить аргумент для директории сохранения результатов
    parser.add_argument('--project', default=ROOT / 'runs/predict-cls', help='save results to project/name')
    # Добавить аргумент для имени сохранения результатов
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # Добавить аргумент для разрешения использования существующей директории
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # Добавить аргумент для использования половинной точности (FP16)
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # Добавить аргумент для использования OpenCV DNN для вывода ONNX моделей
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # Добавить аргумент для шага обработки видеокадров
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # Разобрать аргументы командной строки
    opt = parser.parse_args()
    # Увеличить размер изображения, если он задан только одной величиной
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    # Вывести аргументы командной строки
    print_args(vars(opt))
    return opt


def main(opt):
    # Проверить требования к зависимостям
    check_requirements(exclude=('tensorboard', 'thop'))
    # Запустить функцию run с аргументами
    run(**vars(opt))


if __name__ == "__main__":
    # Разобрать аргументы командной строки
    opt = parse_opt()
    # Запустить основную функцию
    main(opt)
