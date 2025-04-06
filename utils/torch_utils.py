# YOLOv5 🚀 от Ultralytics, лицензия GPL-3.0
"""
Утилиты PyTorch
"""

import math
import os
import platform
import subprocess
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.general import LOGGER, check_version, colorstr, file_date, git_describe

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

try:
    import thop  # для вычисления FLOPs
except ImportError:
    thop = None

# Подавление предупреждений PyTorch
warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')
warnings.filterwarnings('ignore', category=UserWarning)


def smart_inference_mode(torch_1_9=check_version(torch.__version__, '1.9.0')):
    # Применяет декоратор torch.inference_mode(), если torch>=1.9.0, иначе torch.no_grad()
    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate


def smartCrossEntropyLoss(label_smoothing=0.0):
    # Возвращает nn.CrossEntropyLoss с включенным label smoothing для torch>=1.10.0
    if check_version(torch.__version__, '1.10.0'):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        LOGGER.warning(f'WARNING ⚠️ label smoothing {label_smoothing} требует torch>=1.10.0')
    return nn.CrossEntropyLoss()


def smart_DDP(model):
    # Создание DDP модели с проверками
    assert not check_version(torch.__version__, '1.12.0', pinned=True), \
        'torch==1.12.0 torchvision==0.13.0 DDP обучение не поддерживается из-за известной проблемы. ' \
        'Пожалуйста, обновите или понизьте версию torch. См. https://github.com/ultralytics/yolov5/issues/8395'
    if check_version(torch.__version__, '1.11.0'):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


def reshape_classifier_output(model, n=1000):
    # Обновление модели классификации TorchVision до количества классов 'n', если необходимо
    from models.common import Classify
    name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # последний модуль
    if isinstance(m, Classify):  # голова YOLOv5 Classify()
        if m.linear.out_features != n:
            m.linear = nn.Linear(m.linear.in_features, n)
    elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
        if m.out_features != n:
            setattr(model, name, nn.Linear(m.in_features, n))
    elif isinstance(m, nn.Sequential):
        types = [type(x) for x in m]
        if nn.Linear in types:
            i = types.index(nn.Linear)  # индекс nn.Linear
            if m[i].out_features != n:
                m[i] = nn.Linear(m[i].in_features, n)
        elif nn.Conv2d in types:
            i = types.index(nn.Conv2d)  # индекс nn.Conv2d
            if m[i].out_channels != n:
                m[i] = nn.Conv2d(m[i].in_channels, n, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Декоратор, заставляющий все процессы в распределенном обучении ждать, пока локальный мастер что-то сделает
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def device_count():
    # Возвращает количество доступных CUDA-устройств. Безопасная версия torch.cuda.device_count(). Поддерживает Linux и Windows
    assert platform.system() in ('Linux', 'Windows'), 'device_count() поддерживается только на Linux или Windows'
    try:
        cmd = 'nvidia-smi -L | wc -l' if platform.system() == 'Linux' else 'nvidia-smi -L | find /c /v ""'  # Windows
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1])
    except Exception:
        return 0


def select_device(device='', batch_size=0, newline=True):
    # device = None или 'cpu' или 0 или '0' или '0,1,2,3'
    s = f'YOLOv5 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # преобразование в строку, 'cuda:0' в '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # принудительно установить torch.cuda.is_available() = False
    elif device:  # запрошено неготовое к CPU устройство
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # установить переменную среды - должно быть перед assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Недопустимый CUDA '--device {device}' запрошен, используйте '--device cpu' или передайте допустимые CUDA устройства"

    if not cpu and not mps and torch.cuda.is_available():  # предпочтительнее GPU, если доступно
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # количество устройств
        if n > 1 and batch_size > 0:  # проверить, кратно ли batch_size количеству устройств
            assert batch_size % n == 0, f'размер пакета {batch_size} не кратен количеству GPU {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # байты в MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # предпочтительнее MPS, если доступно
        s += 'MPS\n'
        arg = 'mps'
    else:  # вернуться к CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)


def time_sync():
    # Точное время PyTorch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, ops, n=10, device=None):
    """ Профилировщик скорости/памяти/FLOPs YOLOv5
    Использование:
        input = torch.randn(16, 3, 640, 640)
        m1 = lambda x: x * torch.sigmoid(x)
        m2 = nn.SiLU()
        profile(input, [m1, m2], n=100)  # профилирование за 100 итераций
    """
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m  # устройство
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # нет метода backward
                        # print(e)  # для отладки
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list' for x in (x, y))  # формы
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # параметры
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def is_parallel(model):
    # Возвращает True, если модель является типом DP или DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # Депараллелизация модели: возвращает одно-GPU модель, если модель является типом DP или DDP
    return model.module if is_parallel(model) else model


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Находит индексы слоев, соответствующие классу модуля 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Возвращает глобальную редкость модели
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Прореживание модели до запрашиваемой глобальной редкости
    import torch.nn.utils.prune as prune
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # прореживание
            prune.remove(m, 'weight')  # сделать постоянным
    LOGGER.info(f'Модель прорежена до {sparsity(model):.3g} глобальной редкости')


def fuse_conv_and_bn(conv, bn):
    # Слияние слоев Conv2d() и BatchNorm2d() https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Подготовка фильтров
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Подготовка пространственного смещения
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, imgsz=640):
    # Информация о модели. img_size может быть int или список, например img_size=640 или img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # количество параметров
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # количество градиентов
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # максимальный stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # входное изображение в формате BCHW
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # расширить, если int/float
        fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'  # 640x640 GFLOPs
    except Exception:
        fs = ''

    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # Масштабирование img(bs,3,y,x) с учетом ограничения на кратность gs
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # новый размер
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # изменение размера
    if not same_shape:  # заполнить/обрезать img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # значение = среднее значение ImageNet


def copy_attr(a, b, include=(), exclude=()):
    # Копирование атрибутов из b в a, с возможностью включения [...] и исключения [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv5 3-групповой оптимизатор: 0) веса с decadence, 1) веса без decadence, 2) смещения без decadence
    g = [], [], []  # группы параметров оптимизатора
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # слои нормализации, например BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == 'bias':  # смещение (без decadence)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # вес (без decadence)
                g[1].append(p)
            else:
                g[0].append(p)  # вес (с decadence)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # изменить beta1 на momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Оптимизатор {name} не реализован.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # добавить g0 с weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # добавить g1 (веса BatchNorm2d)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) с группами параметров "
                f"{len(g[1])} вес(decay=0.0), {len(g[0])} вес(decay={decay}), {len(g[2])} смещение")
    return optimizer


def smart_hub_load(repo='ultralytics/yolov5', model='yolov5s', **kwargs):
    # Оболочка для torch.hub.load() с умной обработкой ошибок и проблем
    if check_version(torch.__version__, '1.9.1'):
        kwargs['skip_validation'] = True  # валидация вызывает ошибки лимита API GitHub
    if check_version(torch.__version__, '1.12.0'):
        kwargs['trust_repo'] = True  # аргумент требуется начиная с torch 0.12
    try:
        return torch.hub.load(repo, model, **kwargs)
    except Exception:
        return torch.hub.load(repo, model, force_reload=True, **kwargs)


def smart_resume(ckpt, optimizer, ema=None, weights='yolov5s.pt', epochs=300, resume=True):
    # Возобновление обучения из частично обученного чекпоинта
    best_fitness = 0.0
    start_epoch = ckpt['epoch'] + 1
    if ckpt['optimizer'] is not None:
        optimizer.load_state_dict(ckpt['optimizer'])  # оптимизатор
        best_fitness = ckpt['best_fitness']
    if ema and ckpt.get('ema'):
        ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
        ema.updates = ckpt['updates']
    if resume:
        assert start_epoch > 0, f'{weights} обучение до {epochs} эпох завершено, нечего возобновлять.\n' \
                                f"Начните новое обучение без --resume, например 'python train.py --weights {weights}'"
        LOGGER.info(f'Возобновление обучения из {weights} с {start_epoch} эпохи до {epochs} общих эпох')
    if epochs < start_epoch:
        LOGGER.info(f"{weights} обучено на {ckpt['epoch']} эпох. Дополнительная настройка на {epochs} эпох.")
        epochs += ckpt['epoch']  # дополнительная настройка на дополнительные эпохи
    return best_fitness, start_epoch, epochs


class EarlyStopping:
    # Простой Early Stopper для YOLOv5
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # например, mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # количество эпох ожидания после остановки улучшения fitness
        self.possible_stop = False  # возможная остановка может произойти на следующей эпохе

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 для учета начального этапа обучения с нулевым значением fitness
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # количество эпох без улучшения
        self.possible_stop = delta >= (self.patience - 1)  # возможная остановка может произойти на следующей эпохе
        stop = delta >= self.patience  # остановить обучение, если превышено ожидание
        if stop:
            LOGGER.info(f'Ранняя остановка обучения, так как не наблюдается улучшений в течение последних {self.patience} эпох. '
                        f'Лучшие результаты достигнуты на эпохе {self.best_epoch}, лучшая модель сохранена как best.pt.\n'
                        f'Чтобы обновить EarlyStopping(patience={self.patience}) передайте новое значение ожидания, '
                        f'например, `python train.py --patience 300` или используйте `--patience 0`, чтобы отключить EarlyStopping.')
        return stop


class ModelEMA:
    """ Обновленный Exponential Moving Average (EMA) из https://github.com/rwightman/pytorch-image-models
    Поддерживает скользящее среднее всех элементов state_dict модели (параметров и буферов)
    Для деталей EMA см. https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Создание EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # количество обновлений EMA
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # экспоненциальная функция затухания (для помощи в начальных эпохах)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Обновление параметров EMA
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # state_dict модели
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # верно для FP16 и FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} и модель {msd[k].dtype} должны быть FP32'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Обновление атрибутов EMA
        copy_attr(self.ema, model, include, exclude)