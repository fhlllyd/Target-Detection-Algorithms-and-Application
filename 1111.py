import cv2
import numpy as np

# Чтение изображения с диска
image = cv2.imread(r'C:\Users\Lenovo\Desktop\1024.png')

# Преобразование изображения в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Бинаризация изображения (преобразование в черно-белое изображение)
_, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

# Определение структурных элементов (ядер) для морфологических операций
kernel1 = np.ones((10, 10), np.uint8)  # Ядро для операции закрытия
kernel2 = np.ones((3, 3), np.uint8)    # Ядро для операции эрозии
kernel3 = np.ones((2, 2), np.uint8)    # Ядро для операции расширения

# Закрытие (Morphological Closing) — операция, которая сначала выполняет расширение, а затем эрозию.
# Это помогает закрыть маленькие дыры в объектах.
binary_image_closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel1)

# Эрозия — уменьшение размера объектов на изображении.
# Используется для удаления шума или разделения объектов.
binary_image_closed_eroded = cv2.erode(binary_image_closed, kernel2, iterations=25)

# Расширение (Dilation) — увеличение размера объектов на изображении.
# Используется для соединения разорванных частей объектов или восстановления после эрозии.
binary_image_dilated = cv2.dilate(binary_image_closed_eroded, kernel3, iterations=20)

# Повторное закрытие для улучшения качества изображения (если необходимо)
binary_image_closed = cv2.morphologyEx(binary_image_dilated, cv2.MORPH_CLOSE, kernel1)

# Применение медианного фильтра для сглаживания изображения и удаления оставшегося шума
median_filtered_image1 = cv2.medianBlur(binary_image_closed, 5)
median_filtered_image2 = cv2.medianBlur(median_filtered_image1, 5)
median_filtered_image3 = cv2.medianBlur(median_filtered_image2, 5)
median_filtered_image4 = cv2.medianBlur(median_filtered_image3, 5)
median_filtered_image5 = cv2.medianBlur(median_filtered_image4, 5)

# Отображение исходного изображения и обработанных результатов
cv2.imshow('Original', image)
cv2.imshow('Binary Closed Eroded and Dilated', binary_image_closed)
cv2.imshow('Median Filtered', median_filtered_image5)

# Сохранение результата на диск
#cv2.imwrite(r'C:\Users\Lenovo\Desktop\Binary_Closed_Eroded_and_Dilated.png', binary_image_closed)
cv2.imwrite(r'C:\Users\Lenovo\Desktop\Median_Filtered.png', median_filtered_image3)

# Ожидание нажатия клавиши и закрытие всех окон
cv2.waitKey(0)
cv2.destroyAllWindows()