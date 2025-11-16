"""
NVFP4 Batched GEMV - FINAL SUPER-OPTIMIZED VERSION
На основе анализа кода лидеров

Ключевые изменения:
1. Правильный reshape масштабов (возможно НЕ flatten)
2. Оптимальный порядок dimensions для cuBLAS
3. Минимизация промежуточных операций
4. Специальная обработка для разных L
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
    СУПЕР-ОПТИМИЗИРОВАННАЯ реализация на основе анализа лидеров

    Критические оптимизации:
    -------------------------
    1. Правильный формат масштабов для torch._scaled_mm
    2. Минимизация reshape операций
    3. Оптимальный memory layout
    4. Специализация под разные размеры L
    """
    a_ref, b_ref, _, _, sfa_perm, sfb_perm, c_ref = data

    M, K_packed, L = a_ref.shape

    # sfa_perm shape: (32, 4, rest_m, 4, rest_k, L)
    # где rest_m = M/128, rest_k = K/64

    # cuBLAS ожидает: [rest_m, rest_k, 32, 4, 4]
    # У нас:          [32, 4, rest_m, 4, rest_k, L]
    # Нужно: permute(2, 4, 0, 1, 3) -> [rest_m, rest_k, 32, 4, 4]

    # ========================================================================
    # ULTRA FAST PATH: L = 1
    # ========================================================================
    if L == 1:
        # Извлекаем для l=0 и делаем правильный permute
        # [32, 4, rest_m, 4, rest_k] -> [rest_m, rest_k, 32, 4, 4]
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).contiguous()
        scale_b = sfb_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).contiguous()

        # Flatten для _scaled_mm
        scale_a = scale_a.flatten()
        scale_b = scale_b.flatten()

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
    # FAST PATH: 2 <= L <= 8
    # ========================================================================
    if L <= 8:
        # Предварительная обработка всех масштабов
        scales_a = []
        scales_b = []

        for l_idx in range(L):
            # Правильный permute для cuBLAS формата
            sa = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).contiguous().flatten()
            sb = sfb_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).contiguous().flatten()
            scales_a.append(sa)
            scales_b.append(sb)

        # Теперь быстрый цикл только с GPU операциями
        for l_idx in range(L):
            result = torch._scaled_mm(
                a_ref[:, :, l_idx],
                b_ref[:, :, l_idx].transpose(0, 1),
                scales_a[l_idx],
                scales_b[l_idx],
                bias=None,
                out_dtype=torch.float16
            )

            c_ref[:, 0, l_idx] = result[:, 0]

        return c_ref

    # ========================================================================
    # STANDARD PATH: L > 8
    # ========================================================================
    for l_idx in range(L):
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
