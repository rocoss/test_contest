Description

Implement an inclusive prefix sum (scan) kernel that matches the reference implementation.
The kernel should compute the cumulative sum of all elements up to each position.
Because of numerical instability, the tolerance is scaled by the square root of the input size.

Input:
- `data`: A 1D tensor of size `n`
Output:
- `output`: A 1D tensor of size `n`
