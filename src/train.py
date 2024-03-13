import torch
from torch import optim, nn
from .models.clip_finetune import CLIPFineTuneModel, загрузить_и_заморозить_CLIP
from .data.prepare_dataset import CustomImageDataset