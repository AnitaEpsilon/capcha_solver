from PIL import Image
import os
from tqdm import tqdm
from typing import Optional, Any, List

def split_and_save_image(filename: str, input_image_path: str, output_folder_path: str, save_to_tmp: Optional[bool] = False) -> str:
    """
    Разделяет изображение на части и сохраняет их в указанную директорию.

    Аргументы:
    - filename: Имя файла изображения.
    - input_image_path: Путь к директории с исходным изображением.
    - output_folder_path: Путь к директории, куда будут сохранены части изображения.
    - save_to_tmp: Флаг, указывающий на сохранение во временную папку. По умолчанию False.

    Возвращает:
    - Имя папки, в которую были сохранены части изображения.
    """
    # Open the input image
    with Image.open(os.path.join(input_image_path, filename)) as img:

        # Get the filename and the extension
        extension = os.path.basename(filename).split('.')[-1]

        if save_to_tmp:
            foldername = "tmp"
        else:
            foldername = filename.split('.')[0]

        # Define the size of each individual image, assuming they are of equal height
        # and the last image occupies the whole width of the original image
        width, height = img.size
        single_image_height = height // 2  # Divide by 2 because there are 2 rows
        single_image_width = width // 5  # Divide by 5 for the images in the first row
        
        # Create the output directory
        output_directory = os.path.join(output_folder_path, foldername)
        os.makedirs(output_directory, exist_ok=True)
        
        # Split the images and save them
        for i in range(5):  # For the first row
            left = i * single_image_width
            right = (i + 1) * single_image_width
            box = (left, 0, right, single_image_height)
            part_img = img.crop(box)
            part_img.save(os.path.join(output_directory, f'x_{i+1}.{extension}'))
        
        # Save the last image
        box = (0, single_image_height, width*(2/15), 2 * single_image_height)
        part_img = img.crop(box)
        part_img.save(os.path.join(output_directory, f'y.{extension}'))

        return foldername
    
    