"""
NVFP4 Batched GEMV - HYBRID v4

ОБЪЕДИНЯЕТ ЛУЧШЕЕ ИЗ УСПЕШНЫХ ВЕРСИЙ:

От batched_experimental.py (48.529 μs):
✅ Batched processing подход
✅ Кеширование transpose в dict
✅ Простота и чистота кода

От optimized_v3.py (не тестировали, но хорошие идеи):
✅ Pre-computation scale factors для L≤8
✅ Оптимизированные memory operations

НОВОЕ в v4:
✨ torch.compile mode="reduce-overhead" (вместо reduce-overhead)
✨ Dynamic batch_size tuning
✨ Минимизация dict операций для малых L

ОЖИДАЕМЫЙ РЕЗУЛЬТАТ: 45-47 μs (улучшение на 3-7% от 48.529 μs)
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
    Гибридная версия v4 - объединяет лучшие идеи
    """
    a_ref, b_ref, _, _, sfa_perm, sfb_perm, c_ref = data
    M, K_packed, L = a_ref.shape

    # ═══════════════════════════════════════════════════════════
    # FAST PATH: L=1 (самый частый случай)
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
    # OPTIMIZED PATH: L <= 8 (малые батчи)
    # ═══════════════════════════════════════════════════════════
    if L <= 8:
        # КЛЮЧЕВАЯ ОПТИМИЗАЦИЯ: Pre-compute ВСЕХ данных
        scales_a = []
        scales_b = []
        b_t = []

        for l_idx in range(L):
            # Scale factors
            sa = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3)
            sb = sfb_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3)
            scales_a.append(sa.flatten())
            scales_b.append(sb.flatten())

            # Transpose
            b_t.append(b_ref[:, :, l_idx].transpose(0, 1))

        # Минимальный overhead цикл
        for l_idx in range(L):
            result = torch._scaled_mm(
                a_ref[:, :, l_idx],
                b_t[l_idx],
                scales_a[l_idx],
                scales_b[l_idx],
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, 0, l_idx] = result[:, 0]

        return c_ref

    # ═══════════════════════════════════════════════════════════
    # BATCHED PATH: L > 8 (большие батчи)
    # ═══════════════════════════════════════════════════════════
    # Для больших L используем batched processing с кешированием

    # Кеш transpose (это быстро и выгодно)
    b_t_cache = {}
    for l_idx in range(L):
        b_t_cache[l_idx] = b_ref[:, :, l_idx].transpose(0, 1)

    # Обработка по группам для locality
    # Dynamic batch_size в зависимости от L
    batch_size = 8 if L <= 32 else 16

    for batch_start in range(0, L, batch_size):
        batch_end = min(batch_start + batch_size, L)

        # Обрабатываем группу
        for l_idx in range(batch_start, batch_end):
            # On-the-fly scale preparation (для больших L экономим память)
            scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).flatten()
            scale_b = sfb_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).flatten()

            result = torch._scaled_mm(
                a_ref[:, :, l_idx],
                b_t_cache[l_idx],
                scale_a,
                scale_b,
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, 0, l_idx] = result[:, 0]

    return c_ref


# ═══════════════════════════════════════════════════════════════
# torch.compile с max-autotune для более агрессивной оптимизации
# ═══════════════════════════════════════════════════════════════
custom_kernel = torch.compile(
    custom_kernel_impl,
    mode="reduce-overhead",  # Более агрессивные оптимизации
    fullgraph=False,
    dynamic=False,
)
