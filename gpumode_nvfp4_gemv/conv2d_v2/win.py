from task import input_t, output_t
import torch
import torch.nn.functional as F
from utils import DeterministicContext


def custom_kernel(data: input_t) -> output_t:
    """
    Реализация двумерной свертки с использованием PyTorch без отступов.
    Аргументы:
        данные: Кортеж (входной тензор, тензор ядра)
        спецификация: Спецификации свертки
    Возвращается:
        Выходной тензор после свертки
    """
    with DeterministicContext():
        input_tensor, kernel, output = data
        output[...] = F.conv2d(input_tensor, kernel, stride=1, padding=0)
        return output
