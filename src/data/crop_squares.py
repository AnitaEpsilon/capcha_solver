import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
from PIL import Image, ImageDraw
from typing import Optional, Any, List
import os


def open_image_and_visualize(image_path: str, visualize: Optional[bool] = False) -> np.ndarray:
    """
    Открывает изображение по указанному пути и при необходимости визуализирует его.

    Аргументы:
    - image_path: Путь к изображению.
    - visualize: Флаг, определяющий необходимость визуализации. По умолчанию False.

    Возвращает:
    - Массив изображения в формате numpy.ndarray.
    """
    im = cv2.imread(image_path)
    if visualize:
        plt.imshow(im)
        plt.colorbar()
        plt.show()
    return im


def make_scalar_product_mask(im: np.ndarray, visualize: Optional[bool] = False) -> np.ndarray:
    """
    Создает маску изображения на основе скалярного произведения и при необходимости визуализирует её.

    Аргументы:
    - im: Исходное изображение в формате numpy.ndarray.
    - visualize: Флаг, определяющий необходимость визуализации. По умолчанию False.

    Возвращает:
    - Маску изображения в формате numpy.ndarray.
    """
    image = im/255.
    unit_vector = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
    unit_image = image/np.sqrt(np.tile(np.sum(image*image, axis=2), (3,1,1)).transpose(1,2,0))
    new_im = np.sum(unit_image*unit_vector, axis=2) * 255
    if visualize:
        plt.imshow(new_im)
        plt.colorbar()
        plt.show()
    return new_im


def make_binar(im: np.ndarray, threshold: Optional[int] = 240, visualize: Optional[bool] = False) -> np.ndarray:
    """
    Бинаризует изображение по заданному порогу и при необходимости визуализирует его.

    Аргументы:
    - im: Исходное изображение в формате numpy.ndarray.
    - threshold: Порог для бинаризации. По умолчанию 240.
    - visualize: Флаг, определяющий необходимость визуализации. По умолчанию False.

    Возвращает:
    - Бинаризованное изображение в формате numpy.ndarray.
    """
    img = deepcopy(im)
    img[img!=img] = 255
    img[img<threshold] = 0
    img[img>=threshold] = 255
    if visualize:
        plt.imshow(img, cmap="gray")
        plt.colorbar()
        plt.show()
    return img


def pad_borders(im: np.ndarray, pad_width: Optional[int] = 3, visualize: Optional[bool] = False) -> np.ndarray:
    """
    Добавляет рамку вокруг изображения и при необходимости визуализирует его.

    Аргументы:
    - im: Исходное изображение в формате numpy.ndarray.
    - pad_width: Ширина рамки. По умолчанию 3.
    - visualize: Флаг, определяющий необходимость визуализации. По умолчанию False.

    Возвращает:
    - Изображение с рамкой в формате numpy.ndarray.
    """
    im = np.pad(im, 10, 'linear_ramp', end_values=255)
    if visualize:
        plt.imshow(im, cmap='gray')
        plt.colorbar()
        plt.show()
    return im


def smooth_circle_borders(im: np.ndarray, visualize: Optional[bool] = False) -> np.ndarray:
    """
    Сглаживает границы кругов на изображении и при необходимости визуализирует результат.

    Аргументы:
    - im: Исходное изображение в формате numpy.ndarray.
    - visualize: Флаг, определяющий необходимость визуализации. По умолчанию False.

    Возвращает:
    - Изображение с сглаженными границами в формате numpy.ndarray.
    """    
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=2)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, iterations=1)
    if visualize:
        plt.imshow(im, cmap='gray')
        plt.colorbar()
        plt.show()
    return im


def detect_circles(im: np.ndarray, visualize: Optional[bool] = False) -> list:
    """
    Обнаруживает круги на изображении и при необходимости визуализирует результат.

    Аргументы:
    - im: Исходное изображение в формате numpy.ndarray.
    - visualize: Флаг, определяющий необходимость визуализации. По умолчанию False.

    Возвращает:
    - Список центров обнаруженных кругов.
    """
    im = np.uint8(im)
    radius_tolerance = 5
    min_radius = 20 - radius_tolerance
    max_radius = 20 + radius_tolerance

    dp = 1.5  # The inverse ratio of the accumulator resolution to the image resolution
    minDist = 25  # Minimum distance between the centers of the detected circles
    param1 = 40  # The higher threshold of the two passed to the Canny edge detector
    param2 = 18  # Accumulator threshold for the circle centers at the detection stage

    # Use HoughCircles to detect circles
    detected_circles = cv2.HoughCircles(
        im, 
        cv2.HOUGH_GRADIENT, 
        dp, 
        minDist, 
        param1=param1, 
        param2=param2, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )

    # Convert the circle parameters a, b and r to integers.
    detected_circles_rounded = np.uint16(np.around(detected_circles))
    
    # Visualize the results on the new image
    if visualize:
        output_image_circles = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    centers = []
    # Draw the detected circles
    if detected_circles_rounded is not None:
        for i in detected_circles_rounded[0, :]:
            center = (i[0], i[1])
            centers.append(center)

            if visualize:
                radius = i[2]
                # Draw the outer circle
                cv2.circle(output_image_circles, center, radius, (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(output_image_circles, center, 2, (0, 0, 255), 3)

    if visualize:
        plt.imshow(output_image_circles)
        plt.colorbar()
        plt.show()

    return centers


def substract_padding(centers: list, padding: Optional[int] = 10) -> list:
    """
    Корректирует координаты центров кругов, учитывая добавленную рамку.

    Аргументы:
    - centers: Список кортежей с координатами центров кругов.
    - padding: Ширина добавленной рамки. По умолчанию 10.

    Возвращает:
    - Список скорректированных координат центров кругов.
    """
    new_centers = []
    for c in centers:
        new_centers.append((c[0]-padding, c[1]-padding))
    return new_centers


def sort_centers(centers: list, img_path: str, visualize: Optional[bool] = False) -> list:
    """
    Сортирует центры кругов по их расстоянию до шестой точки и при необходимости визуализирует результат.

    Аргументы:
    - centers: Список кортежей с координатами центров кругов.
    - img_path: Путь к изображению.
    - visualize: Флаг, определяющий необходимость визуализации. По умолчанию False.

    Возвращает:
    - Список отсортированных координат центров кругов.
    """
    # Define the coordinates of the 6th point
    sixth_point = (40, 190)

    # Calculate distances from the sixth point to each of the centers
    distances = [np.sqrt((x - sixth_point[0])**2 + (y - sixth_point[1])**2) for x, y in centers]

    # Pair each center with its distance from the sixth point
    center_distances = list(zip(centers, distances))

    # Sort the centers by their distance from the sixth point
    sorted_centers = sorted(center_distances, key=lambda x: x[1])

    # Extract the sorted centers and their order
    sorted_centers_only = [center for center, distance in sorted_centers]

    if visualize:
        # Load the provided image
        img = Image.open(img_path)
        # Convert to RGB to plot color on top of the original image
        img_rgb = img.convert('RGB')
        draw = ImageDraw.Draw(img_rgb)

        # Draw the sixth point
        draw.ellipse((sixth_point[0]-3, sixth_point[1]-3, sixth_point[0]+3, sixth_point[1]+3), fill='blue', outline='blue')

        # Draw the centers, lines, and order
        for num, (x, y) in enumerate(sorted_centers_only):
            # Draw the center
            draw.ellipse((x-3, y-3, x+3, y+3), fill='red', outline='red')
            # Draw line from the center to the sixth point
            draw.line((x, y, sixth_point[0], sixth_point[1]), fill='green', width=1)
            # Annotate the order next to the center
            draw.text((x+5, y), f'{num+1}', fill='purple',)

        plt.imshow(img_rgb)
        plt.colorbar()
        plt.show()

    return sorted_centers_only


def define_corners(center: tuple, radius: Optional[int] = 15) -> tuple:
    """
    Определяет координаты углов квадрата вокруг центра круга с заданным радиусом.

    Аргументы:
    - center: Координаты центра круга.
    - radius: Радиус квадрата. По умолчанию 15.

    Возвращает:
    - Кортеж с координатами углов квадрата.
    """
    left = center[0] - radius
    right = center[0] + radius
    up = center[1] - radius
    down = center[1] + radius
    
    return (left, up, right, down)


def get_last_directory(path: str) -> str:
    """
    Возвращает имя последней директории в указанном пути.

    Аргументы:
    - path: Путь, из которого нужно извлечь имя последней директории.

    Возвращает:
    - Имя последней директории в пути.
    """
    directory_path = os.path.dirname(path)
    return os.path.basename(directory_path)



def crop_squares(image_path: str, squares: list, output_dir: str, visualize: Optional[bool] = False):
    """
    Вырезает квадраты из изображения по заданным координатам и сохраняет их.

    Аргументы:
    - image_path: Путь к изображению.
    - squares: Список кортежей, каждый из которых представляет координаты квадрата.
    - output_dir: Директория для сохранения вырезанных квадратов.
    - visualize: Флаг, определяющий необходимость визуализации. По умолчанию False.
    """
    subfoldername = os.path.basename(image_path).split('.')[0]
    foldername = get_last_directory(image_path)
    full_path = os.path.join(output_dir,foldername,subfoldername)
    os.makedirs(full_path, exist_ok=True)

    # Load the image
    img = Image.open(image_path)

    # Convert the image to RGB if it's not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    if visualize:
        draw = ImageDraw.Draw(img)

    # Draw each square
    for i, (left, up, right, down) in enumerate(squares):
        crop = img.crop((left, up, right, down))
        crop_path = os.path.join(full_path, f"{i+1}.png")
        crop.save(crop_path, quality=95)
    
    if visualize:
        for i, (left, up, right, down) in enumerate(squares):
            # Draw a rectangle for each square
            draw.rectangle([(left, up), (right, down)], outline='yellow', width=2)
        plt.imshow(img)
        plt.colorbar()
        plt.show()


def combine_functions_to_get_centers(image_path: str, visualize: Optional[bool] = False) -> list:
    """
    Комбинирует несколько функций для обработки изображения и получения центров кругов.

    Аргументы:
    - image_path: Путь к изображению.
    - visualize: Флаг, определяющий необходимость визуализации. По умолчанию False.

    Возвращает:
    - Список координат центров кругов.
    """
    im = open_image_and_visualize(image_path, visualize=visualize)
    im = make_scalar_product_mask(im, visualize=visualize)
    im = make_binar(im, threshold=240)
    im = pad_borders(im, pad_width=10, visualize=visualize)
    im = smooth_circle_borders(im, visualize=visualize)
    centers = detect_circles(im, visualize=visualize)
    centers = substract_padding(centers, padding=10)
    centers = sort_centers(centers, image_path, visualize=visualize)
    return centers


def save_squares(image_path: str, output_dir: str, centers: list, visualize: Optional[bool] = False):
    """
    Сохраняет квадраты, вырезанные вокруг центров кругов, в указанную директорию.

    Аргументы:
    - image_path: Путь к изображению.
    - output_dir: Директория для сохранения вырезанных квадратов.
    - centers: Список координат центров кругов.
    - visualize: Флаг, определяющий необходимость визуализации. По умолчанию False.
    """
    squares = [define_corners(center, radius=20) for center in centers]
    crop_squares(image_path, squares, output_dir, visualize=visualize)




def median(values: List[float]) -> Optional[float]:
    """
    Вычисляет медиану списка значений.

    Аргументы:
    - values: Список значений.

    Возвращает:
    - Медианное значение списка. Возвращает None, если список пуст.
    """
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        return None
    if n % 2 == 1:
        return sorted_values[n // 2]
    else:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2

def calculate_median_coordinates(list_of_lists: List[List[tuple]]) -> List[tuple]:
    """
    Вычисляет медианные координаты для списка списков координат.

    Аргументы:
    - list_of_lists: Список списков координат.

    Возвращает:
    - Список медианных координат.
    """
    median_coordinates = []
    for i in range(5):  # Предполагаем, что требуется 5 пар координат
        x_values = []
        y_values = []
        for lst in list_of_lists:
            if i < len(lst):  # Убедимся, что индекс не выходит за пределы списка
                x_values.append(lst[i][0])
                y_values.append(lst[i][1])
        median_x = median(x_values)
        median_y = median(y_values)
        if median_x and median_y:
            median_coordinates.append((median_x, median_y))
    return median_coordinates