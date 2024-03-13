import easyocr
import os
from PIL import Image


# Создание объекта reader
reader = easyocr.Reader(['en'], gpu=False,)  # Укажите 'gpu=True', если доступна GPU

# Функция для распознавания текста с изображения
def recognize_digit(image_path: str) -> str:
    """
    Распознает цифру на изображении с использованием EasyOCR.

    Аргументы:
    - image_path: Путь к изображению для распознавания.

    Возвращает:
    - Распознанную цифру в виде строки. В случае отсутствия распознавания возвращает "1".
    """
    result = reader.readtext(image_path, detail=1, paragraph=False, allowlist='12345LlIiAa!')

    if not result:
        return "1"
    for detection in result:
        bbox, text, confidence = detection
        if text[0] in "Aa":
            return "4"
        
        if text[0] in "LlIi!":
            return "1"
        
        return text[0]


def save_bottom_half(input_folder: str, output_folder: str) -> None:
    """
    Вырезает и сохраняет нижнюю половину изображения 'y.jpg' из входной папки в выходную папку под именем 'im.jpg'.

    Аргументы:
    - input_folder: Папка, содержащая исходное изображение 'y.jpg'.
    - output_folder: Папка для сохранения результата.

    Возвращает:
    - None
    """
    # Open the input image

    input_image_path = os.path.join(input_folder, "y.jpg")
    output_image_path = os.path.join(output_folder, "im.jpg")
    os.makedirs(output_folder, exist_ok=True)

    with Image.open(input_image_path) as img:
        width, height = img.size
        
        # Calculate the box for the bottom half
        # The box is defined by (left, upper, right, lower) edges.
        box = (0, height // 2, width, height)
        
        # Crop the bottom half
        bottom_half = img.crop(box)
        
        # Save the cropped image to the specified output path
        bottom_half.save(output_image_path)