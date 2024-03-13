from typing import Any
from matplotlib import pyplot as plt
import os
import json
from PIL import Image
import matplotlib.patches as patches

def display_comparison(base_image: Any, compare_image: Any, similarity: float, i: int) -> None:
    """
    Отображает сравнение базового изображения с изображением для сравнения, включая показатель сходства.

    Аргументы:
    - base_image: Изображение для отображения как базовое.
    - compare_image: Изображение для сравнения с базовым.
    - similarity: Показатель сходства между изображениями.
    - i: Индекс изображения для сравнения.

    Возвращает:
    - None
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(base_image)
    plt.title("Base Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(compare_image)
    plt.title(f"Compare to Image {i}\nSimilarity: {similarity.item():.4f}")
    plt.axis('off')

    plt.show()


def visualize_labels(name: str, answer: tuple = None, path: str = "data/raw") -> None:
    """
    Визуализирует метки на изображении с использованием данных из JSON-файла.
    
    Параметры:
        name (str): Имя файла без расширения, используется для загрузки соответствующего изображения и JSON-файла.
        answer (tuple, optional): Координаты точки (x, y), которая будет отмечена на изображении как ответ. По умолчанию None.
        path (str): Путь к директории, содержащей изображения и JSON-файлы. По умолчанию "data/raw".
    
    Возвращает:
        None: Функция только визуализирует изображение с нанесенными метками и не возвращает никаких значений.
    
    Функция загружает изображение и соответствующий ему JSON-файл, содержащий данные о формах (например, прямоугольники и точки),
    которые должны быть отображены на изображении. Затем она отображает изображение и наносит на него указанные формы с метками.
    Для прямоугольников рисуются контуры с метками посередине, а для точек - круги с метками. Если указан ответ, он также отображается
    на изображении в виде круга другого цвета.
    """
    
    # Загрузка данных из JSON
    with open(os.path.join(path, f'{name}.json')) as json_file:
        data = json.load(json_file)

    # Загрузка изображения
    image_path = os.path.join(path, f'{name}.jpg')
    image = Image.open(image_path)

    # Инициализация фигуры для отображения
    fig, axes = plt.subplots(2, figsize=(10, 8))
    plt.title(name)
    axes[0].imshow(image)
    axes[1].imshow(image)

    # Обработка и отображение форм из JSON
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            top_left = shape['points'][0]
            bottom_right = shape['points'][1]
            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]
            rect = patches.Rectangle((top_left[0], top_left[1]), width, height, linewidth=2, edgecolor='r', facecolor='none')
            axes[1].add_patch(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width()/2.0
            cy = ry + rect.get_height()/2.0
            axes[1].annotate(shape['label'], (cx, cy), color='purple', weight='bold', fontsize=8, ha='center', va='center')
        elif shape['shape_type'] == 'point':
            circle = patches.Circle((shape['points'][0][0], shape['points'][0][1]), radius=10, color='y')
            axes[1].add_patch(circle)
            axes[1].annotate(shape['label'], (shape['points'][0][0], shape['points'][0][1]), color='w', weight='bold', fontsize=8, ha='center', va='center')

    # Отображение ответа, если он задан
    if answer is not None:
        circle = patches.Circle(answer, radius=5, color='b')
        axes[1].add_patch(circle)

    # Скрытие осей для лучшей визуализации
    axes[0].axis('off')
    axes[1].axis('off')
    
    plt.show()  # Показать результат
