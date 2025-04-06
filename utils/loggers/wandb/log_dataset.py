import argparse

from wandb_utils import WandbLogger

from utils.general import LOGGER

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def create_dataset_artifact(opt):
    # Создаем объект WandbLogger для работы с Weights & Biases, указывая тип задания "Dataset Creation"
    # Обратите внимание, что возвращаемое значение здесь не используется (TODO: return value unused)
    logger = WandbLogger(opt, None, job_type='Dataset Creation')
    if not logger.wandb:
        # Если модуль wandb не доступен, выводим сообщение о том, чтобы установить wandb с помощью pip
        LOGGER.info("install wandb using `pip install wandb` to log the dataset")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Путь к файлу конфигурации датасета (например, data.yaml)
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    # Флаг для обучения как одно классного датасета
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    # Имя проекта Weights & Biases
    parser.add_argument('--project', type=str, default='YOLOv5', help='name of W&B Project')
    # Сущность Weights & Biases (возможно, организация или пользователь)
    parser.add_argument('--entity', default=None, help='W&B entity')
    # Имя выполнения Weights & Biases (run)
    parser.add_argument('--name', type=str, default='log dataset', help='name of W&B run')

    opt = parser.parse_args()
    # Явно запрещаем проверку на возобновление для задания по загрузке датасета
    opt.resume = False

    create_dataset_artifact(opt)