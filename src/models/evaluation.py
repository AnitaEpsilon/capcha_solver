import json
from math import sqrt
from typing import Optional, Any, List
import os
from tqdm import tqdm
from PIL import Image
import torch

from src.data.preprocessing import split_and_save_image
from src.data.preprocess_labels import recognize_digit, save_bottom_half
from src.models.clip_finetune import CLIPFineTuneModel
from src.data.crop_squares import save_squares, combine_functions_to_get_centers, calculate_median_coordinates
from src.visualization.visualize import display_comparison



def find_point_and_rect(json_file_path: str) -> tuple:
    """
    Находит координаты точки и прямоугольника, соответствующие условиям задачи, основываясь на данных из JSON файла.

    Аргументы:
    - json_file_path: Путь к JSON файлу с аннотациями.

    Возвращает:
    - Кортеж, содержащий координаты точки, номер точки, координаты прямоугольника и смещение прямоугольника по оси X.
    """
    # Загрузка данных из файла
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    digit_labels = [(shape['label'], max(point[1] for point in shape['points'])) for shape in data['shapes'] if shape['label'].isdigit()]
    special_number, _ = max(digit_labels, key=lambda x: x[1])
    ordinary_points = [shape for shape in data['shapes'] if shape['label'] == '+']
    icons = [shape for shape in data['shapes'] if shape['label'] == 'icon']
    point_a = (40, 190) 
    
    def conditional_distance(point_a, point_b):
        x_b_mod = point_b[0] % 200
        distance = sqrt((x_b_mod - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)
        return distance

    ordinary_points_info = [{
        'coords': shape['points'][0],
        'distance': conditional_distance(point_a, shape['points'][0]),
        'picture_number': int(shape['points'][0][0] // 200) + 1
    } for shape in ordinary_points]

    ordinary_points_info.sort(key=lambda x: x['distance'])
    
    for index, point in enumerate(ordinary_points_info, start=1):
        point['length_number'] = index
        
    matching_point = next((p for p in ordinary_points_info if p['length_number'] == int(special_number)), None)
    

    def is_point_in_rect(point, rect):
        return rect[0][0] <= point[0] <= rect[1][0] and rect[0][1] <= point[1] <= rect[1][1]

    def find_shift_and_rect(point, icons):
        for icon in icons:
            for shift_x in range(0, 10001, 200):  # Предполагаем, что смещение может быть до 10000 пикселей
                shifted_rect = [[x + shift_x, y] for x, y in icon['points']]
                if is_point_in_rect(point, shifted_rect):
                    return shifted_rect, shift_x

        return None, None

    # Находим подходящий прямоугольник и требуемое смещение
    matching_rect, shift_x = find_shift_and_rect(matching_point['coords'], icons)

    return matching_point['coords'], matching_point['length_number'], matching_rect, shift_x


def center_for_answer(centers: list, idx: int, digit: int) -> list:
    """
    Рассчитывает центр для ответа на основе индекса и распознанной цифры.

    Аргументы:
    - centers: Список центров.
    - idx: Индекс выбранного изображения.
    - digit: Распознанная цифра.

    Возвращает:
    - Центр для ответа в виде списка координат.
    """
    width = 200
    center = list(centers[digit-1])
    center[0] = center[0] + (idx-1) * width
    return center


def is_point_in_rectangle(point: tuple, rectangle: tuple) -> bool:
    """
    Проверяет, находится ли точка внутри заданного прямоугольника.

    Аргументы:
    - point: Координаты точки (x, y).
    - rectangle: Координаты прямоугольника ((x1, y1), (x2, y2)).

    Возвращает:
    - True, если точка находится внутри прямоугольника, иначе False.
    """
    x, y = point
    x1, y1 = rectangle[0]
    x2, y2 = rectangle[1]
    
    if (x >= x1 and x <= x2) and (y >= y1 and y <= y2):
        return True
    else:
        return False
    

def compare_images(base_image_folder: str, compare_image_folder: str, digit: int, finetune_model, preprocess, visualize: bool = False) -> int:
    """
    Функция для сравнения изображений и определения наиболее похожего.
    
    :param base_image_folder: Путь к папке с базовым изображением.
    :param compare_image_folder: Путь к папке с изображениями для сравнения.
    :param digit: Цифра для определения номера сравниваемого изображения.
    :param finetune_model: Модель для сравнения изображений.
    :param preprocess: Функция предварительной обработки изображений.
    :param visualize: Флаг для визуализации процесса сравнения (по умолчанию False).
    
    :return: Индекс наиболее похожего изображения.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_image_path = os.path.join(base_image_folder, "im.jpg")
    base_image_raw = Image.open(base_image_path).convert("RGB").resize((300, 300))
    base_image = preprocess(base_image_raw).unsqueeze(0).to(device)

    max_similarity = -1
    max_similarity_index = -1

    for i in range(1, 6):
        compare_image_path = os.path.join(compare_image_folder, f"x_{i}", f"{digit}.png")   #jpg")
        compare_image_raw = Image.open(compare_image_path).convert("RGB").resize((300, 300))
        compare_image = preprocess(compare_image_raw).unsqueeze(0).to(device)
        try:
            #compare_features, compare_image = get_image_features(compare_image_path, clip_model, preprocess)
            # Использование модели надстройки для сравнения
            similarity = finetune_model(base_image, compare_image) #base_features, compare_features).cpu().numpy()
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = i
            if visualize:
                display_comparison(base_image_raw, compare_image_raw, similarity, i)
        except FileNotFoundError:
            print(f"Файл {compare_image_path} не найден.")

    return max_similarity_index



def calculate_metrics(finetune_model: CLIPFineTuneModel,
                      preprocess,
                      input_image_path: str = "data/raw",
                      output_folder_path: str = "data/interim/orbits",
                      output_for_objects_path: str = "data/interim/objects",
                      path_to_chosen_files: Optional[str] = None,
                      visualize: bool = False,) -> None:
    """
    Вычисляет метрики для набора изображений.

    Аргументы:
    - finetune_model: Модель для сравнения изображений
    - preprocess: функция подготовки изображений для подачи в модель
    - input_image_path: Путь к входным изображениям.
    - output_folder_path: Путь к промежуточным результатам.
    - output_for_objects_path: Путь к обработанным объектам.
    - path_to_chosen_files: Путь к файлу со списком выбранных файлов для обработки.
    - visualize: Производить ли визуализацию.

    Возвращает:
    - error_files: Список сэмплов, на которых предсказание было ложным.
    - error_answers: Список соответствующих списку error_files ложно предсказанных координат.
    """

    is_correct = []
    files = os.listdir(input_image_path)

    if path_to_chosen_files:
        with open(path_to_chosen_files, 'r') as file:
            valid_names = file.read().split("\n")
            jpg_files = [file for file in files if file.split('.')[0] in valid_names and file.endswith('.jpg')]
    else:
        jpg_files = [file for file in files if file.endswith('.jpg')]
    error_files = []
    error_answers = []
    except_cnt = 0
    for file in tqdm(jpg_files):
        try:
        #if 1 == 1:
            foldername = split_and_save_image(file, input_image_path, output_folder_path, save_to_tmp=True)
            name_without_format = file.split(".")[0]
            digit = recognize_digit(os.path.join(output_folder_path, foldername, "y.jpg"))
            save_bottom_half(input_folder=os.path.join(output_folder_path, foldername),
                            output_folder=os.path.join(output_for_objects_path, foldername))
            list_of_centers = []
            for i in range(1, 6):
                image_path = os.path.join(output_folder_path, foldername, f"x_{i}.jpg")
                centers = combine_functions_to_get_centers(image_path, visualize=visualize)
                list_of_centers.append(centers)

            centers = calculate_median_coordinates(list_of_centers)
            for i in range(1, 6):
                image_path = os.path.join(output_folder_path, foldername, f"x_{i}.jpg")
                save_squares(image_path, output_for_objects_path, centers, visualize=visualize)        

            idx = compare_images(base_image_folder=os.path.join(output_for_objects_path, foldername), 
                                compare_image_folder=os.path.join(output_for_objects_path, foldername), 
                                digit=digit, 
                                finetune_model=finetune_model, 
                                preprocess=preprocess,
                                visualize=visualize)
            
            matching_point, number, correct_square, shift_x = find_point_and_rect(json_file_path=os.path.join(input_image_path, f"{name_without_format}.json"))
            answer =  center_for_answer(centers, idx, int(digit))
            is_ok = int(is_point_in_rectangle(point=answer, rectangle=correct_square))
            
            if not is_ok:
                error_files.append(file)
                error_answers.append(answer)
            is_correct.append(is_ok)
        except:
            is_correct.append(0)
            except_cnt += 1
        
    print(f"Number of evaluated samples: {len(is_correct)}")
    print(f"Accuracy: {sum(is_correct)/len(is_correct)}")
    print(f"Accuracy without exceptions: {sum(is_correct)/(len(is_correct)-except_cnt)}")    

    return error_files, error_answers
