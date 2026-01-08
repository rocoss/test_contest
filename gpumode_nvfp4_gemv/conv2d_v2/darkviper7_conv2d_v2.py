import torch
import triton
import triton.language as tl
from utils import make_match_reference, DeterministicContext
from task import input_t, output_t

# --- Reference Implementation (Provided by you) ---
import torch.nn.functional as F

def ref_kernel(data: input_t) -> output_t:
    with DeterministicContext():
        input_tensor, kernel, output = data
        return F.conv2d(
            input_tensor,
            kernel,
            stride=1,
            padding=0,
        )

# --- Triton Implementation ---

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'loop_unroll_factor': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64, 'loop_unroll_factor': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 32, 'loop_unroll_factor': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'loop_unroll_factor': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64, 'loop_unroll_factor': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 32, 'loop_unroll_factor': 4}, num_warps=4, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'IN_H', 'IN_W', 'K_H', 'K_W']
)
@triton.jit
def _conv2d_triton(
    # Pointers
    input_ptr, weight_ptr, output_ptr,
    # Dimensions
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    K_H, K_W,
    # Strides
    stride_input_b, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_k, stride_weight_c, stride_weight_h, stride_weight_w,
    stride_output_b, stride_output_k, stride_output_h, stride_output_w,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
    num_stages: tl.constexpr,
    loop_unroll_factor: tl.constexpr
):
    # PID mapping
    pid_m = tl.program_id(0) # Spatial + Batch
    pid_n = tl.program_id(1) # Output Channels

    # Calculate M (Total Spatial Elements) and N (Output Channels)
    M = BATCH * OUT_H * OUT_W
    N = OUT_C

    # Create offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Coordinate reconstruction from flattened M
    # index = b * (OH * OW) + y * OW + x
    out_x = offs_m % OUT_W
    remain = offs_m // OUT_W
    out_y = remain % OUT_H
    batch = remain // OUT_H

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over Input Channels (c) and Kernel Dimensions (r, s)
    for c in range(IN_C):
        for r in range(K_H):
            for s in tl.range(K_W,num_stages=num_stages, loop_unroll_factor=loop_unroll_factor):
                # Weights: [Out_C, In_C, K_H, K_W]
                # Load a [1, BLOCK_N] row of weights
                w_offset = (offs_n * stride_weight_k) + (c * stride_weight_c) + (r * stride_weight_h) + (s * stride_weight_w)
                w = tl.load(weight_ptr + w_offset, mask=mask_n, other=0.0)

                # Input: [Batch, In_C, In_H, In_W]
                # We need pixel (batch, c, out_y + r, out_x + s)
                in_y = out_y + r
                in_x = out_x + s
                
                # Load a [BLOCK_M, 1] column of inputs
                in_offset = (batch * stride_input_b) + (c * stride_input_c) + (in_y * stride_input_h) + (in_x * stride_input_w)
                val = tl.load(input_ptr + in_offset, mask=mask_m, other=0.0)

                # Accumulate: Outer product [BLOCK_M, 1] * [1, BLOCK_N]
                accumulator += val[:, None] * w[None, :]

    # Store Result
    # Output: [Batch, Out_C, Out_H, Out_W]
    # We need to construct pointers for the [BLOCK_M, BLOCK_N] block
    output_base = (batch * stride_output_b) + (out_y * stride_output_h) + (out_x * stride_output_w)
    output_offset = output_base[:, None] + (offs_n * stride_output_k)[None, :]
    
    tl.store(output_ptr + output_offset, accumulator, mask=mask_m[:, None] & mask_n[None, :])


def custom_kernel(data: input_t) -> output_t:
    """
    Wrapper to bridge the competition input format with the Triton Kernel.
    """
    input_tensor, kernel, output_tensor = data
    
    # 1. Extract Shapes
    B, IN_C, IN_H, IN_W = input_tensor.shape
    OUT_C, _, K_H, K_W = kernel.shape
    OUT_H = output_tensor.shape[2]
    OUT_W = output_tensor.shape[3]
    
    # 2. Calculate Grid
    # We flatten Batch, Height, and Width into one dimension 'M'
    total_output_elements = B * OUT_H * OUT_W
    
    grid = lambda META: (
        triton.cdiv(total_output_elements, META['BLOCK_SIZE_M']),
        triton.cdiv(OUT_C, META['BLOCK_SIZE_N'])
    )
    
    # 3. Launch Kernel
    _conv2d_triton[grid](
        input_tensor, kernel, output_tensor,
        B, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        K_H, K_W,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        kernel.stride(0), kernel.stride(1), kernel.stride(2), kernel.stride(3),
        output_tensor.stride(0), output_tensor.stride(1), output_tensor.stride(2), output_tensor.stride(3),
    )
    
    return output_tensor

if __name__ == "__main__":
    # This runs the comparison against the reference kernel
    check_implementation = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-3)
    check_implementation(solution_kernel)
