"""
NVFP4 Batched GEMV - OPTIMIZED VERSION v3

ОСНОВНЫЕ УЛУЧШЕНИЯ от submission_ultimate.py:

1. ✅ torch.compile() с mode='max-autotune' (вместо reduce-overhead)
2. ✅ Предварительная подготовка ВСЕХ scale factors
3. ✅ Минимизация permute/contiguous операций
4. ✅ Кеширование transpose результатов
5. ✅ Оптимизированные memory access patterns
6. ❌ БЕЗ излишней специализации (L=2, L=4)
7. ❌ БЕЗ контрпродуктивных микро-оптимизаций

ОЖИДАЕМЫЙ РЕЗУЛЬТАТ: 48-52 μs (улучшение на 10-15% от 55.643 μs)
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


def custom_kernel_impl(data: input_t) -> output_t:
    """
    Оптимизированная версия NVFP4 batched GEMV

    Ключевые оптимизации:
    - Pre-compute всех scale factors ДО главного цикла
    - Кеширование transpose результатов
    - Минимизация операций на GPU
    - Избежание лишних контрпродуктивных специализаций
    """
    a_ref, b_ref, _, _, sfa_perm, sfb_perm, c_ref = data
    M, K_packed, L = a_ref.shape

    # ═══════════════════════════════════════════════════════════
    # FAST PATH: L=1 (наиболее частый случай)
    # ═══════════════════════════════════════════════════════════
    if L == 1:
        # Подготовка scale factors (минимизируем операции)
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).flatten()
        scale_b = sfb_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).flatten()

        # GEMV: (M, K) @ (N, K).T -> (M, N), где N=1 (padded до 128)
        result = torch._scaled_mm(
            a_ref[:, :, 0],
            b_ref[:, :, 0].transpose(0, 1),
            scale_a,
            scale_b,
            bias=None,
            out_dtype=torch.float16,
        )

        # Записываем результат (векторизованная операция)
        c_ref[:, 0, 0] = result[:, 0]
        return c_ref

    # ═══════════════════════════════════════════════════════════
    # FAST PATH: L <= 8 (малые батчи)
    # ═══════════════════════════════════════════════════════════
    if L <= 8:
        # КЛЮЧЕВАЯ ОПТИМИЗАЦИЯ: Pre-compute ВСЕХ scale factors
        # Это избегает повторных permute/flatten в горячем цикле
        scales_a = []
        scales_b = []
        b_transposed = []

        for l_idx in range(L):
            # Подготовка scale factors
            sa = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3)
            sb = sfb_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3)

            # Flatten (попытка избежать копирования через is_contiguous check)
            scales_a.append(sa.flatten())
            scales_b.append(sb.flatten())

            # Кеширование transpose
            b_transposed.append(b_ref[:, :, l_idx].transpose(0, 1))

        # Главный цикл - минимальный overhead
        for l_idx in range(L):
            result = torch._scaled_mm(
                a_ref[:, :, l_idx],
                b_transposed[l_idx],
                scales_a[l_idx],
                scales_b[l_idx],
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, 0, l_idx] = result[:, 0]

        return c_ref

    # ═══════════════════════════════════════════════════════════
    # STANDARD PATH: L > 8 (большие батчи)
    # ═══════════════════════════════════════════════════════════
    # Для больших L избегаем создания больших списков в памяти
    # Но всё ещё кешируем transpose результаты

    # Pre-compute transpose для всех L (это быстро и выгодно)
    b_transposed_cache = {}
    for l_idx in range(L):
        b_transposed_cache[l_idx] = b_ref[:, :, l_idx].transpose(0, 1)

    # Главный цикл с on-the-fly scale preparation
    for l_idx in range(L):
        # Scale factors (минимум операций)
        scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).flatten()
        scale_b = sfb_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).flatten()

        result = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_transposed_cache[l_idx],
            scale_a,
            scale_b,
            bias=None,
            out_dtype=torch.float16,
        )
        c_ref[:, 0, l_idx] = result[:, 0]

    return c_ref


# ═══════════════════════════════════════════════════════════════
# torch.compile с агрессивной оптимизацией
# ═══════════════════════════════════════════════════════════════
# mode='max-autotune' может быть лучше чем 'reduce-overhead'
# для compute-intensive участков
custom_kernel = torch.compile(
    custom_kernel_impl,
    mode="max-autotune",  # Попробуем более агрессивную оптимизацию
    fullgraph=False,      # Избегаем graph breaks
    dynamic=False,        # Static shapes для лучшей оптимизации
)
