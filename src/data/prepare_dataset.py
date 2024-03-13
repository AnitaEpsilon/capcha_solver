import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

class CustomImageDataset(Dataset):
    """
    Класс датасета для пар изображений с метками их соответствия.

    Атрибуты:
    - dataframe (pandas.DataFrame): DataFrame с путями к изображениям и метками соответствия.
    - transform (callable, optional): Преобразование, применяемое к каждому изображению.
    - base_path (str): Базовый путь к изображениям.

    Методы:
    - __len__(): Возвращает размер датасета.
    - __getitem__(idx): Возвращает пару изображений и метку их соответствия по индексу.
    """

    def __init__(self, dataframe, transform=None, base_path='train_data2/benchmarked/'):
        """
        Инициализирует датасет с указанным DataFrame, трансформацией и базовым путем.

        Параметры:
        - dataframe (pandas.DataFrame): DataFrame с данными.
        - transform (callable, optional): Функция преобразования изображений. По умолчанию None.
        - base_path (str): Базовый путь к изображениям. По умолчанию 'train_data2/benchmarked/'.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.base_path = base_path

    def __len__(self):
        """
        Возвращает общее количество пар изображений в датасете.

        Выходные данные:
        - int: Количество пар изображений.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Возвращает пару изображений и метку их соответствия по указанному индексу.

        Параметры:
        - idx (int): Индекс пары изображений в датасете.

        Выходные данные:
        - tuple: Пара трансформированных изображений и метка соответствия (torch.Tensor, torch.Tensor, float).
        """
        img1_path = self.base_path + self.dataframe.iloc[idx, 0]
        img2_path = self.base_path + self.dataframe.iloc[idx, 1]
        label = self.dataframe.iloc[idx, 2].astype(np.float32)

        img1 = Image.open(img1_path).convert("RGB").resize((300, 300))
        img2 = Image.open(img2_path).convert("RGB").resize((300, 300))

        if self.transform:
            img1 = self.transform(img1)
            img1 = (img1 - torch.min(img1)) / (torch.max(img1) - torch.min(img1))
            img2 = self.transform(img2)
            img2 = (img2 - torch.min(img2)) / (torch.max(img2) - torch.min(img2))

        return img1, img2, label