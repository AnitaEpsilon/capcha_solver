import torch
from torch import nn
import clip

class CLIPFineTuneModel(nn.Module):
    """
    Класс для настройки модели CLIP с добавлением линейного слоя для финетюнинга.

    Атрибуты:
    - clip_model (torch.nn.Module): предобученная модель CLIP.
    - linear (torch.nn.Linear): линейный слой для трансформации признаков.
    - cosine_similarity (torch.nn.CosineSimilarity): слой для вычисления косинусного сходства между векторами признаков.

    Методы:
    - forward(img1, img2): Проход вперед, который принимает пары изображений и возвращает оценку их сходства.
    """

    def __init__(self, clip_model):
        """
        Инициализирует класс с предобученной моделью CLIP и добавляет линейный слой.

        Параметры:
        - clip_model (torch.nn.Module): предобученная модель CLIP.
        """
        super(CLIPFineTuneModel, self).__init__()
        self.clip_model = clip_model
        self.linear = nn.Linear(512, 64)
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, img1, img2):
        """
        Выполняет проход вперед, принимая два изображения и вычисляя их сходство.

        Параметры:
        - img1 (torch.Tensor): тензор первого изображения.
        - img2 (torch.Tensor): тензор второго изображения.

        Выходные данные:
        - similarity (torch.Tensor): тензор с оценками сходства изображений.
        """
        with torch.no_grad():
            features1 = self.clip_model.encode_image(img1).to(dtype=torch.float32)
            features2 = self.clip_model.encode_image(img2).to(dtype=torch.float32)

        transformed_features1 = self.linear(features1).to(dtype=torch.float32)
        transformed_features2 = self.linear(features2).to(dtype=torch.float32)
        transformed_features1 = transformed_features1 / transformed_features1.norm(dim=-1, keepdim=True)
        transformed_features2 = transformed_features2 / transformed_features2.norm(dim=-1, keepdim=True)

        similarity = self.cosine_similarity(transformed_features1, transformed_features2).unsqueeze(-1).to(dtype=torch.float32)
        return similarity
    

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
finetune_model = CLIPFineTuneModel(model).to(device)
model_path = 'models/finetuned_clip_model_roc09997.pth'

finetune_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))