# YOLOv5 🚀 от Ultralytics, лицензия GPL-3.0
"""
Запуск YOLOv5 для детекции объектов на изображениях, видео, директориях, вебкамерах, потоках и т.д.

Использование - источники:
    $ python detect.py --weights yolov5s.pt --source 0                               # вебкамера
                                                     img.jpg                         # изображение
                                                     vid.mp4                         # видео
                                                     screen                          # скриншот
                                                     path/                           # директория
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP поток

Использование - форматы:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime или OpenCV DNN с --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (только macOS)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Корневой каталог YOLOv5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Добавляем ROOT в PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Относительный путь

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # Путь к модели или Triton URL
        source=ROOT / 'data/images',  # Файл/директория/URL/glob/скриншот/0(вебкамера)
        data=ROOT / 'data/coco128.yaml',  # Путь к dataset.yaml
        imgsz=(640, 640),  # Размер изображения для инференса (высота, ширина)
        conf_thres=0.25,  # Порог уверенности
        iou_thres=0.45,  # Порог IoU для NMS
        max_det=1000,  # Максимальное количество детекций на изображении
        device='',  # Устройство CUDA, например, 0 или 0,1,2,3 или cpu
        view_img=False,  # Показывать результаты
        save_txt=False,  # Сохранять результаты в *.txt
        save_conf=False,  # Сохранять уверенность в метках --save-txt
        save_crop=False,  # Сохранять обрезанные предсказанные рамки
        nosave=False,  # Не сохранять изображения/видео
        classes=None,  # Фильтровать по классам
        agnostic_nms=False,  # Класс-независимая NMS
        augment=False,  # Улучшенный инференс
        visualize=False,  # Визуализировать характеристики
        update=False,  # Обновить все модели
        project=ROOT / 'runs/detect',  # Сохранить результаты в project/name
        name='exp',  # Сохранить результаты в project/name
        exist_ok=False,  # Допускается существующий project/name, не увеличивать
        line_thickness=3,  # Толщина рамки (пиксели)
        hide_labels=False,  # Скрыть метки
        hide_conf=False,  # Скрыть уверенность
        half=False,  # Использовать FP16 для инференса с полуплавающей точностью
        dnn=False,  # Использовать OpenCV DNN для ONNX инференса
        vid_stride=1,  # Шаг между кадрами видео
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # Сохранять изображения с результатами
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # Скачать файл

    # Директории
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # Увеличить номер запуска
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # Создать директорию

    # Загрузить модель
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # Проверить размер изображения

    # Загрузчик данных
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

    # Выполнить инференс
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # Разогрев
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 в fp16/32
            im /= 255  # 0 - 255 в 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # Добавить размер батча

        # Инференс
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Вторичный классификатор (опционально)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Обработка предсказаний
        for i, det in enumerate(pred):  # На каждое изображение
            seen += 1
            if webcam:  # Размер батча >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # Преобразовать в Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # Строка вывода
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Коэффициент нормализации whwh
            imc = im0.copy() if save_crop else im0  # Для save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Масштабирование рамок с img_size на im0 размер
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Вывод результатов
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # Количество детекций для класса
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Добавить к строке

                # Запись результатов
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Запись в файл
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # Нормализованные xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # Формат метки
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Добавить рамку к изображению
                        c = int(cls)  # Целочисленный класс
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Поток результатов
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # Разрешить изменение размера окна (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 миллисекунда

            # Сохранение результатов (изображение с детекциями)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' или 'stream'
                    if vid_path[i] != save_path:  # Новое видео
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # Освободить предыдущий видеозаписывающий объект
                        if vid_cap:  # Видео
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # Поток
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # Принудительно использовать суффикс *.mp4 для результирующих видео
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Вывод времени (только инференс)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Вывод результатов
    t = tuple(x.t / seen * 1E3 for x in dt)  # Скорость на изображение
    LOGGER.info(f'Скорость: %.1fмс предобработка, %.1fмс инференс, %.1fмс NMS на изображении размером {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} меток сохранено в {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Результаты сохранены в {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # Обновить модель (для исправления предупреждения SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'C:\Users\Lenovo\Desktop\文件夹\yolov5-7.0\yolov5-7.0\yolov5s-seg.pt', help='путь к модели или Triton URL')
    parser.add_argument('--source', type=str, default=r'C:\Users\Lenovo\Desktop\1.png', help='файл/директория/URL/glob/скриншот/0(вебкамера)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(опционально) путь к dataset.yaml')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='размер изображения h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='порог уверенности')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='порог IoU для NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='максимальное количество детекций на изображении')
    parser.add_argument('--device', default='', help='устройство CUDA, например, 0 или 0,1,2,3 или cpu')
    parser.add_argument('--view-img', action='store_true', help='показывать результаты')
    parser.add_argument('--save-txt', action='store_true', help='сохранять результаты в *.txt')
    parser.add_argument('--save-conf', action='store_true', help='сохранять уверенность в метках --save-txt')
    parser.add_argument('--save-crop', action='store_true', help='сохранять обрезанные предсказанные рамки')
    parser.add_argument('--nosave', action='store_true', help='не сохранять изображения/видео')
    parser.add_argument('--classes', nargs='+', type=int, help='фильтровать по классам')
    parser.add_argument('--agnostic-nms', action='store_true', help='класс-независимая NMS')
    parser.add_argument('--augment', action='store_true', help='улучшенный инференс')
    parser.add_argument('--visualize', action='store_true', help='визуализировать характеристики')
    parser.add_argument('--update', action='store_true', help='обновить все модели')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='сохранить результаты в project/name')
    parser.add_argument('--name', default='exp', help='сохранить результаты в project/name')
    parser.add_argument('--exist-ok', action='store_true', help='допускается существующий project/name, не увеличивать')
    parser.add_argument('--line-thickness', default=3, type=int, help='толщина рамки (пиксели)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='скрыть метки')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='скрыть уверенность')
    parser.add_argument('--half', action='store_true', help='использовать FP16 для инференса с полуплавающей точностью')
    parser.add_argument('--dnn', action='store_true', help='использовать OpenCV DNN для ONNX инференса')
    parser.add_argument('--vid-stride', type=int, default=1, help='шаг между кадрами видео')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # Расширить
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)