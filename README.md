# NumFlatTensorAdapter

NumFlatTensorAdapter bridges the NumFlat numerical library with `.NET`'s official `System.Numerics.Tensors` APIs.

## What it does

- Converts NumFlat-specific types (`Vec<T>` and `Mat<T>`) into `TensorSpan<T>`.
- Makes it easier to apply highly optimized tensor operations from Microsoft to NumFlat-based computations.

## Why use it

By connecting NumFlat data structures to `TensorSpan<T>`, you can keep NumFlat's programming model while taking advantage of the performance-focused implementation in `System.Numerics.Tensors`.
