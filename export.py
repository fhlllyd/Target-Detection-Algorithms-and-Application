# YOLOv5 🚀 от Ultralytics, лицензия GPL-3.0
"""
Экспорт модели YOLOv5 PyTorch в другие форматы. Экспорты TensorFlow написаны https://github.com/zldrobit

Формат                      | `export.py --include`         | Модель
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/
PaddlePaddle                | `paddle`                      | yolov5s_paddle_model/

Требования:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Использование:
    $ python export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...

Вывод:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime или OpenCV DNN с --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (только для macOS)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import contextlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

# Определение путей к файлам и директориям
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Корневая директория YOLOv5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Добавление ROOT в PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Относительный путь

# Импорт модулей и функций
from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_version,
                           check_yaml, colorstr, file_size, get_default_args, print_args, url2file, yaml_save)
from utils.torch_utils import select_device, smart_inference_mode

# Определение переменной для проверки macOS
MACOS = platform.system() == 'Darwin'  # macOS окружение

# Функция для получения таблицы поддерживаемых форматов экспорта
def export_formats():
    # Форматы экспорта YOLOv5
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
        ['TensorFlow.js', 'tfjs', '_web_model', False, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True],]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])

# Декоратор для обработки ошибок экспорта
def try_export(inner_func):
    # Декоратор экспорта YOLOv5, например @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} экспорт успешен ✅ {dt.t:.1f}s, сохранено как {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} экспорт неудачен ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func

# Функция для экспорта в формат TorchScript
@try_export
def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    # Экспорт модели YOLOv5 в формат TorchScript
    LOGGER.info(f'\n{prefix} начало экспорта с torch {torch.__version__}...')
    f = file.with_suffix('.torchscript')

    ts = torch.jit.trace(model, im, strict=False)
    d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
    extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
    if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f, None

# Функция для экспорта в формат ONNX
@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
    # Экспорт модели YOLOv5 в формат ONNX
    check_requirements('onnx')
    import onnx

    LOGGER.info(f'\n{prefix} начало экспорта с onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')

    output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
            dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic только совместим с cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Проверки
    model_onnx = onnx.load(f)  # загрузка модели onnx
    onnx.checker.check_model(model_onnx)  # проверка модели onnx

    # Метаданные
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Упрощение
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
            import onnxsim

            LOGGER.info(f'{prefix} упрощение с onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f'{prefix} ошибка упрощения: {e}')
    return f, model_onnx

# Функция для экспорта в формат OpenVINO
@try_export
def export_openvino(file, metadata, half, prefix=colorstr('OpenVINO:')):
    # Экспорт модели YOLOv5 в формат OpenVINO
    check_requirements('openvino-dev')  # требуется openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.inference_engine as ie

    LOGGER.info(f'\n{prefix} начало экспорта с openvino {ie.__version__}...')
    f = str(file).replace('.pt', f'_openvino_model{os.sep}')

    cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f} --data_type {'FP16' if half else 'FP32'}"
    subprocess.run(cmd.split(), check=True, env=os.environ)  # экспорт
    yaml_save(Path(f) / file.with_suffix('.yaml').name, metadata)  # добавление metadata.yaml
    return f, None

# Функция для экспорта в формат PaddlePaddle
@try_export
def export_paddle(model, im, file, metadata, prefix=colorstr('PaddlePaddle:')):
    # Экспорт модели YOLOv5 в формат PaddlePaddle
    check_requirements(('paddlepaddle', 'x2paddle'))
    import x2paddle
    from x2paddle.convert import pytorch2paddle

    LOGGER