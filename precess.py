# Импортируем библиотеку OpenCV для работы с изображениями
import cv2

# Считываем изображение
# Функция cv2.imread() загружает изображение из указанного пути.
# В качестве параметра передаётся путь к файлу изображения.
# Возвращает изображение в виде массива NumPy.
# Если файл не найден или путь некорректен, функция вернёт None.
image = cv2.imread(r'C:\Users\Lenovo\Desktop\777.png')

# Применяем медианный фильтр
# Функция cv2.medianBlur() выполняет медианную фильтрацию изображения.
# Первый параметр — исходное изображение.
# Второй параметр — размер ядра фильтра (должен быть положительным нечётным числом), здесь указан 5.
# Медианный фильтр — это метод нелинейной фильтрации, который заменяет значение каждого пикселя на медианное значение в его окрестности,
# что позволяет устранить шумы, сохраняя при этом границы объектов.
smoothed_image3 = cv2.medianBlur(image, 5)

# Выводим результат
# Функция cv2.imshow() создаёт окно и отображает в нём фильтрованное изображение.
# Первый параметр — название окна (можно задать любое).
# Второй параметр — массив изображения, которое нужно отобразить.
cv2.imshow('Gaussian Blur', smoothed_image3)
# Функция cv2.waitKey() ожидает нажатия клавиши пользователем.
# Параметр — время ожидания в миллисекундах. Значение 0 означает бесконечное ожидание до нажатия любой клавиши.
cv2.waitKey(0)
# Функция cv2.imwrite() сохраняет фильтрованное изображение в указанном пути.
# Первый параметр — путь сохранения файла.
# Второй параметр — массив изображения, которое нужно сохранить.
cv2.imwrite(r'C:\Users\Lenovo\Desktop\789.png', smoothed_image3)
# Функция cv2.destroyAllWindows() закрывает все окна, созданные с помощью OpenCV.
cv2.destroyAllWindows()