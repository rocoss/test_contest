"""
NVFP4 Batched GEMV - torch.compile() OPTIMIZATION

Это быстрое решение для оптимизации текущего кода.
torch.compile() может дать 5-10% улучшение за счет:
1. Фузии операций
2. Оптимизации памяти
3. Автоматического выбора лучших операций

Ожидаемый результат: 55-58 μs (от 58.948 μs текущего)
Установка: встроено в PyTorch 2.0+
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


def custom_kernel_base(data: input_t) -> output_t:
    """
    Оригинальный базовый kernel из submission_final.py
    """
    a_ref, b_ref, _, _, sfa_perm, sfb_perm, c_ref = data
    M, K_packed, L = a_ref.shape

    # ════════════════════════════════════════════════════════════════════════
    # ULTRA FAST PATH: L = 1
    # ════════════════════════════════════════════════════════════════════════
    if L == 1:
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).contiguous()
        scale_b = sfb_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).contiguous()
        
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

    # ════════════════════════════════════════════════════════════════════════
    # FAST PATH: 2 <= L <= 8
    # ════════════════════════════════════════════════════════════════════════
    if L <= 8:
        scales_a = []
        scales_b = []
        for l_idx in range(L):
            sa = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).contiguous().flatten()
            sb = sfb_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).contiguous().flatten()
            scales_a.append(sa)
            scales_b.append(sb)

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

    # ════════════════════════════════════════════════════════════════════════
    # STANDARD PATH: L > 8
    # ════════════════════════════════════════════════════════════════════════
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


# Компилируем kernel с torch.compile()
# mode="max-autotune" - максимальная оптимизация, но долгая компиляция
# mode="reduce-overhead" - оптимизирует overhead операций (рекомендуется)
try:
    # Попробуем использовать максимальную оптимизацию
    custom_kernel = torch.compile(
        custom_kernel_base,
        mode="reduce-overhead",  # Быстрее компилируется, хороший баланс
        fullgraph=False,         # Не требует полный граф (гибче)
    )
except Exception as e:
    # Если что-то не сработало, используем оригинальный kernel
    print(f"Warning: torch.compile() failed: {e}")
    print("Falling back to original kernel")
    custom_kernel = custom_kernel_base

# Alias
optimized_kernel = custom_kernel
