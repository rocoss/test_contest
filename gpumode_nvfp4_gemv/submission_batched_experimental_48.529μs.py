"""
NVFP4 Batched GEMV - EXPERIMENTAL BATCHED VERSION

РАДИКАЛЬНЫЙ ПОДХОД: Попытка объединить все L в меньшее количество kernel launches

ГИПОТЕЗА: Если обработать несколько L за раз с помощью tensor операций,
          можем снизить kernel launch overhead

РИСКИ: Может быть медленнее из-за дополнительных reshape/stack операций

ОЖИДАНИЕ: 50-60 μs (неопределённо - эксперимент!)
"""

import torch
from typing import Tuple

input_t = Tuple[
    torch.Tensor,  # a: M×K×L
    torch.Tensor,  # b: 1×K×L
    torch.Tensor,  # sfa: M×(K/16)×L (CPU)
    torch.Tensor,  # sfb: 1×(K/16)×L (CPU)
    torch.Tensor,  # sfa_permuted
    torch.Tensor,  # sfb_permuted
    torch.Tensor,  # c: M×1×L
]

output_t = torch.Tensor


def custom_kernel_impl(data: input_t) -> output_t:
    """
    ЭКСПЕРИМЕНТАЛЬНАЯ версия с batched processing

    Идея: Обрабатывать группы L вместе, чтобы уменьшить
          количество kernel launches
    """
    a_ref, b_ref, _, _, sfa_perm, sfb_perm, c_ref = data
    M, K_packed, L = a_ref.shape

    # ═══════════════════════════════════════════════════════════
    # L=1: оптимальный fast path (без изменений)
    # ═══════════════════════════════════════════════════════════
    if L == 1:
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).flatten()
        scale_b = sfb_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).flatten()

        result = torch._scaled_mm(
            a_ref[:, :, 0],
            b_ref[:, :, 0].transpose(0, 1),
            scale_a,
            scale_b,
            bias=None,
            out_dtype=torch.float16,
        )

        c_ref[:, 0, 0] = result[:, 0]
        return c_ref

    # ═══════════════════════════════════════════════════════════
    # BATCHED PROCESSING: Обработка по группам
    # ═══════════════════════════════════════════════════════════
    # Оптимальный размер группы (tunable параметр)
    batch_size = 4 if L <= 8 else 8

    # Кеширование transpose один раз для всех
    b_transposed = {}
    for l_idx in range(L):
        b_transposed[l_idx] = b_ref[:, :, l_idx].transpose(0, 1)

    # Обработка по группам
    for batch_start in range(0, L, batch_size):
        batch_end = min(batch_start + batch_size, L)
        batch_range = range(batch_start, batch_end)

        # Обработка группы последовательно
        # (Попытка batched _scaled_mm может быть добавлена здесь)
        for l_idx in batch_range:
            scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).flatten()
            scale_b = sfb_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).flatten()

            result = torch._scaled_mm(
                a_ref[:, :, l_idx],
                b_transposed[l_idx],
                scale_a,
                scale_b,
                bias=None,
                out_dtype=torch.float16,
            )

            c_ref[:, 0, l_idx] = result[:, 0]

    return c_ref


# Используем reduce-overhead для этой версии (более стабильно)
custom_kernel = torch.compile(
    custom_kernel_impl,
    mode="reduce-overhead",
    fullgraph=False,
    dynamic=False,
)
