using NumFlat;
using System;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace NumFlat
{
    public static class TensorAdapter
    {
        public static TensorSpan<T> AsTensorSpan<T>(this Mat<T> mat) where T : unmanaged, INumberBase<T>
        {
            Nint2 lengths = default;
            lengths[0] = mat.RowCount;
            lengths[1] = mat.ColCount;

            Nint2 strides = default;
            strides[0] = 1;
            strides[1] = mat.Stride;

            return new TensorSpan<T>(mat.Memory.Span, lengths, strides);
        }
    }



    [InlineArray(2)]
    internal struct Nint2
    {
        private nint _element0;
    }

}
