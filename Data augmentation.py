import cv2
import os

def extract_frames(video_path, output_dir, interval=30):
    """
    Извлечение кадров из видео с заданным интервалом.

    :param video_path: Путь к видеофайлу.
    :param output_dir: Директория для сохранения извлеченных кадров.
    :param interval: Интервал между извлеченными кадрами (в кадрах).
    """
    # Создаем директорию для сохранения кадров, если она не существует
    os.makedirs(output_dir, exist_ok=True)

    # Открываем видеофайл с помощью OpenCV
    cap = cv2.VideoCapture(video_path)
    # Проверяем, удалось ли открыть видеофайл
    if not cap.isOpened():
        print("Ошибка: Невозможно открыть видеофайл")
        return

    # Инициализируем счетчики кадров
    count = 0  # Счетчик сохраненных кадров
    frame_count = 0  # Общий счетчик кадров

    # Цикл для чтения кадров из видео
    while True:
        # Считываем очередной кадр
        ret, frame = cap.read()
        # Если кадр не считан, выходим из цикла
        if not ret:
            break

        # Проверяем, нужно ли сохранять текущий кадр
        if frame_count % interval == 0:
            # Формируем имя файла для сохранения кадра
            output_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
            # Сохраняем кадр в указанную директорию
            cv2.imwrite(output_path, frame)
            # Увеличиваем счетчик сохраненных кадров
            count += 1

        # Увеличиваем общий счетчик кадров
        frame_count += 1

    # Освобождаем ресурсы, связанные с видеофайлом
    cap.release()


# Пример использования функции
extract_frames(
    video_path="your_video.mp4",  # Замените на путь к вашему видеофайлу
    output_dir="extracted_frames",  # Директория для сохранения кадров
    interval=30  # Интервал между кадрами (например, 30 кадров, что соответствует 1 кадру в секунду для видео с частотой 30 кадров в секунду)
)