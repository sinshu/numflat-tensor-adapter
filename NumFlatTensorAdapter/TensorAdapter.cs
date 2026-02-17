using NumFlat;
using System;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace NumFlat
{
    /// <summary>
    /// Provides extension methods to view NumFlat vectors and matrices as <see cref="TensorSpan{T}"/>.
    /// </summary>
    public static class TensorAdapter
    {
        /// <summary>
        /// Creates a two-dimensional <see cref="TensorSpan{T}"/> that references the same memory as the matrix.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="mat">The source matrix.</param>
        /// <returns>
        /// A tensor span with shape <c>(<paramref name="mat"/>.RowCount, <paramref name="mat"/>.ColCount)</c>
        /// over the matrix storage.
        /// </returns>
        public static TensorSpan<T> AsTensorSpan<T>(this Mat<T> mat) where T : unmanaged, INumberBase<T>
        {
            Nint2 lengths = default;
            lengths[0] = mat.RowCount;
            lengths[1] = mat.ColCount;

            // TensorSpan does not accept non-zero strides for singleton dimensions
            // when those strides can overlap with another axis.
            // NumFlat matrices can be 1xN / Nx1, so normalize the singleton axis stride to 0.
            Nint2 strides = default;
            strides[0] = mat.RowCount == 1 ? 0 : 1;
            strides[1] = mat.ColCount == 1 ? 0 : mat.Stride;

            return new TensorSpan<T>(mat.Memory.Span, lengths, strides);
        }

        /// <summary>
        /// Creates a one-dimensional <see cref="TensorSpan{T}"/> that references the same memory as the vector.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="vec">The source vector.</param>
        /// <returns>
        /// A tensor span with shape <c>(<paramref name="vec"/>.Count)</c>
        /// over the vector storage.
        /// </returns>
        public static TensorSpan<T> AsTensorSpan<T>(this Vec<T> vec) where T : unmanaged, INumberBase<T>
        {
            nint lengths = vec.Count;

            // TensorSpan does not accept non-zero strides for singleton dimensions
            // when those strides can overlap with another axis.
            nint strides = vec.Count == 1 ? 0 : vec.Stride;

            return new TensorSpan<T>(vec.Memory.Span, [lengths], [strides]);
        }
    }



    [InlineArray(2)]
    internal struct Nint2
    {
        private nint _element0;
    }

}
