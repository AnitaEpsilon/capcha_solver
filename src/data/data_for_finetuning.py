from PIL import Image

def save_correct_answer(rectangle_coords: list, image_path: str, cropped_image_path: str) -> None:
    """
    Вырезает и сохраняет указанный прямоугольник из изображения.

    Аргументы:
    - rectangle_coords: Координаты прямоугольника для вырезки.
    - image_path: Путь к исходному изображению.
    - cropped_image_path: Путь для сохранения вырезанного прямоугольника.

    Возвращает:
    - None
    """
    # Reload the original image
    image = Image.open(image_path)

    # Define the coordinates of the rectangle to be cropped
    top_left_x, top_left_y = rectangle_coords[0]
    bottom_right_x, bottom_right_y = rectangle_coords[1]

    # Crop the image
    cropped_image = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))

    # Save the cropped image to a file
    cropped_image.save(cropped_image_path)



import json


def find_matching_square(path_to_json: str) -> list:
    """
    Находит координаты прямоугольника, содержащего знак плюс, в JSON файле аннотаций.

    Аргументы:
    - path_to_json: Путь к JSON файлу с аннотациями.

    Возвращает:
    - Координаты прямоугольника, содержащего знак плюс, в формате списка из двух точек.
    """
    
    with open(path_to_json) as json_file:
        data = json.load(json_file)
    # First, let's find the coordinates of all pluses
    plus_signs = [shape['points'][0] for shape in data['shapes'] if shape['label'] == '+']

    # Now, we will check which rectangles contain these plus signs
    # A point is inside a rectangle if its x coordinate is between the x coordinates of the rectangle
    # and its y coordinate is between the y coordinates of the rectangle

    def is_point_in_rect(point, rect):
        # unpack points
        px, py = point
        (rx1, ry1), (rx2, ry2) = rect
        return rx1 <= px <= rx2 and ry1 <= py <= ry2

    # List to store rectangles that contain a plus
    rectangles_with_plus = []

    # Iterate over the shapes to find rectangles that contain a plus
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            # Get rectangle coordinates
            rect_coords = shape['points']
            # Check each plus sign
            for plus in plus_signs:
                if is_point_in_rect(plus, rect_coords):
                    return rect_coords

import json

def find_rectangles_without_plus(path_to_json: str) -> list:
    """
    Находит координаты прямоугольников, не содержащих знак плюса, в файле JSON с аннотациями.

    Аргументы:
    - path_to_json: Путь к файлу JSON с аннотациями.

    Возвращает:
    - Список координат для каждого прямоугольника, не содержащего знак плюса.
      Каждый набор координат представлен в формате списка из двух точек.
    """
    
    with open(path_to_json) as json_file:
        data = json.load(json_file)
    
    # Находим координаты всех знаков плюса
    plus_signs = [shape['points'][0] for shape in data['shapes'] if shape['label'] == '+']

    # Функция для проверки, находится ли точка внутри прямоугольника
    def is_point_in_rect(point, rect):
        px, py = point
        (rx1, ry1), (rx2, ry2) = rect
        return rx1 <= px <= rx2 and ry1 <= py <= ry2

    # Список для хранения всех прямоугольников
    all_rectangles = []

    # Список для хранения прямоугольников, содержащих плюс
    rectangles_with_plus = []

    # Сначала получаем все прямоугольники
    for shape in data['shapes']:
        if (shape['shape_type'] == 'rectangle') and not (shape['label'] in ["1", "2", "3", "4", "5"]) :
            all_rectangles.append(shape['points'])

    # Определяем прямоугольники с знаком плюс
    for rect in all_rectangles:
        for plus in plus_signs:
            if is_point_in_rect(plus, rect):
                rectangles_with_plus.append(rect)
                break  # Нет необходимости проверять другие знаки плюс, если один найден

    # Теперь получаем список прямоугольников без знака плюс путем удаления rectangles_with_plus из all_rectangles
    rectangles_without_plus = [rect for rect in all_rectangles if rect not in rectangles_with_plus]

    return rectangles_without_plus