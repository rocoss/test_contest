"""
NVFP4 Batched GEMV - CORRECT OPTIMIZATION v3
На основе анализа: почему safe_ultra был медленнее

Ключевой инсайт: contiguous() на больших тензорах дорого!
Решение: Минимизировать копирования, оптимизировать только горячий путь
"""

import torch
from typing import Tuple

input_t = Tuple[
    torch.Tensor,  # a: M×K×L
    torch.Tensor,  # b: 1×K×L  
    torch.Tensor,  # sfa: M×(K/16)×L (CPU)
    torch.Tensor,  # sfb: 1×(K/16)×L (CPU)
    torch.Tensor,  # sfa_permuted: (32, 4, rest_m, 4, rest_k, L)
    torch.Tensor,  # sfb_permuted: (32, 4, rest_n, 4, rest_k, L)
    torch.Tensor,  # c: M×1×L
]

output_t = torch.Tensor


def custom_kernel(data: input_t) -> output_t:
    """
    ПРАВИЛЬНАЯ оптимизация - учитываем стоимость операций

    Оптимизации:
    1. L=1: Специальный fast path (избегаем overhead цикла)
    2. L>1: Стандартный путь (проверено работает)
    3. НЕТ дорогих contiguous() на больших тензорах
    """
    a_ref, b_ref, _, _, sfa_perm, sfb_perm, c_ref = data

    M, K_packed, L = a_ref.shape

    # ========================================================================
    # CRITICAL OPTIMIZATION: Специальный путь для L=1
    # ========================================================================
    # L=1 встречается в 1/3 бенчмарков и критичен для результата
    if L == 1:
        # Прямой путь без цикла - экономим ~1-2 μs
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).contiguous().flatten()
        scale_b = sfb_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).contiguous().flatten()

        result = torch._scaled_mm(
            a_ref[:, :, 0],
            b_ref[:, :, 0].transpose(0, 1),
            scale_a,
            scale_b,
            bias=None,
            out_dtype=torch.float16
        )

        c_ref[:, 0, 0] = result[:, 0]
        return c_ref

    # ========================================================================
    # STANDARD PATH: L > 1
    # ========================================================================
    # Используем проверенный подход из submission_final
    # Не пытаемся оптимизировать что работает!

    for l_idx in range(L):
        # Маленький срез → contiguous() быстрый
        scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).contiguous().flatten()
        scale_b = sfb_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).contiguous().flatten()

        result = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a,
            scale_b,
            bias=None,
            out_dtype=torch.float16
        )

        c_ref[:, 0, l_idx] = result[:, 0]

    return c_ref


# Alias
optimized_kernel = custom_kernel
