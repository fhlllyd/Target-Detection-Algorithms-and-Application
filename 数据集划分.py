import os
import shutil
import random

random.seed(0)  # Установка фиксированного сид для воспроизводимости


def split_data(file_path, xml_path, new_file_path, train_rate, val_rate, test_rate):
    """
    Разделение датасета на обучающую, валидационную и тестовую выборки.

    Аргументы:
    file_path: Путь к каталогу с изображениями.
    xml_path: Путь к каталогу с метками (аннотациями).
    new_file_path: Путь к каталогу, в который будут сохранены разделенные данные.
    train_rate: Процент данных для обучающей выборки.
    val_rate: Процент данных для валидационной выборки.
    test_rate: Процент данных для тестовой выборки.
    """
    each_class_image = []  # Список для хранения имен изображений
    each_class_label = []  # Список для хранения имен меток
    for image in os.listdir(file_path):
        each_class_image.append(image)  # Добавление имени изображения в список
    for label in os.listdir(xml_path):
        each_class_label.append(label)  # Добавление имени метки в список
    data = list(zip(each_class_image, each_class_label))  # Объединение списков изображений и меток в пары
    total = len(each_class_image)  # Общее количество изображений
    random.shuffle(data)  # Перемешивание данных для случайного разделения
    each_class_image, each_class_label = zip(*data)  # Разделение перемешанных данных обратно на изображения и метки
    train_images = each_class_image[0:int(train_rate * total)]  # Выделение обучающих изображений
    val_images = each_class_image[int(train_rate * total):int((train_rate + val_rate) * total)]  # Выделение валидационных изображений
    test_images = each_class_image[int((train_rate + val_rate) * total):]  # Выделение тестовых изображений
    train_labels = each_class_label[0:int(train_rate * total)]  # Выделение обучающих меток
    val_labels = each_class_label[int(train_rate * total):int((train_rate + val_rate) * total)]  # Выделение валидационных меток
    test_labels = each_class_label[int((train_rate + val_rate) * total):]  # Выделение тестовых меток

    # Копирование обучающих изображений и меток в новый каталог
    for image in train_images:
        print(image)  # Вывод имени изображения
        old_path = file_path + '/' + image  # Старый путь к изображению
        new_path1 = new_file_path + '/' + 'train' + '/' + 'images'  # Новый путь к каталогу с обучающими изображениями
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)  # Создание каталога, если он не существует
        new_path = new_path1 + '/' + image  # Новый путь к изображению
        shutil.copy(old_path, new_path)  # Копирование изображения

    for label in train_labels:
        print(label)  # Вывод имени метки
        old_path = xml_path + '/' + label  # Старый путь к метке
        new_path1 = new_file_path + '/' + 'train' + '/' + 'labels'  # Новый путь к каталогу с обучающими метками
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)  # Создание каталога, если он не существует
        new_path = new_path1 + '/' + label  # Новый путь к метке
        shutil.copy(old_path, new_path)  # Копирование метки

    # Копирование валидационных изображений и меток в новый каталог
    for image in val_images:
        old_path = file_path + '/' + image  # Старый путь к изображению
        new_path1 = new_file_path + '/' + 'val' + '/' + 'images'  # Новый путь к каталогу с валидационными изображениями
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)  # Создание каталога, если он не существует
        new_path = new_path1 + '/' + image  # Новый путь к изображению
        shutil.copy(old_path, new_path)  # Копирование изображения

    for label in val_labels:
        old_path = xml_path + '/' + label  # Старый путь к метке
        new_path1 = new_file_path + '/' + 'val' + '/' + 'labels'  # Новый путь к каталогу с валидационными метками
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)  # Создание каталога, если он не существует
        new_path = new_path1 + '/' + label  # Новый путь к метке
        shutil.copy(old_path, new_path)  # Копирование метки

    # Копирование тестовых изображений и меток в новый каталог
    for image in test_images:
        old_path = file_path + '/' + image  # Старый путь к изображению
        new_path1 = new_file_path + '/' + 'test' + '/' + 'images'  # Новый путь к каталогу с тестовыми изображениями
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)  # Создание каталога, если он не существует
        new_path = new_path1 + '/' + image  # Новый путь к изображению
        shutil.copy(old_path, new_path)  # Копирование изображения

    for label in test_labels:
        old_path = xml_path + '/' + label  # Старый путь к метке
        new_path1 = new_file_path + '/' + 'test' + '/' + 'labels'  # Новый путь к каталогу с тестовыми метками
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)  # Создание каталога, если он не существует
        new_path = new_path1 + '/' + label  # Новый путь к метке
        shutil.copy(old_path, new_path)  # Копирование метки


if __name__ == '__main__':
    file_path = "data/wheel_dataset_3016/wheel_dataset_3016/images"  # Путь к каталогу с изображениями
    xml_path = 'data/wheel_dataset_3016/wheel_dataset_3016/json-txt'  # Путь к каталогу с метками
    new_file_path = "data/wheel_dataset_3016/wheel—data"  # Путь к каталогу для сохранения разделенных данных
    split_data(file_path, xml_path, new_file_path, train_rate=0.7, val_rate=0.1, test_rate=0.2)  # Вызов функции разделения данных