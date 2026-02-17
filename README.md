# NumFlatTensorAdapter

NumFlatTensorAdapter bridges the NumFlat numerical library with .NET's official `System.Numerics.Tensors` APIs.

## Overview

- Converts NumFlat-specific types (`Vec<T>` and `Mat<T>`) into `TensorSpan<T>`.
- Makes it easier to apply highly optimized tensor operations from Microsoft to NumFlat-based computations.

## Installation

.NET 10 is required.

[The NuGet package](https://www.nuget.org/packages/NumFlatTensorAdapter) is available.

```
dotnet add package NumFlatTensorAdapter
```

## Usage

Call the `AsTensorSpan()` method to convert a vector of matrix to `TensorSpan<T>`.

```cs
Mat<double> a =
[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
];

Mat<double> b =
[
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1],
];

var c = new Mat<double>(3, 3);

// Use a TensorSpan method.
Tensor.Add(a.AsTensorSpan(), b.AsTensorSpan(), c.AsTensorSpan());
```

## License

NumFlatTensorAdapter is available under [the MIT license](LICENSE.txt).
