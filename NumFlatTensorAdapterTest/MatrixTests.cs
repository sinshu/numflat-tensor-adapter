using System;
using System.Numerics.Tensors;
using NumFlat;

namespace NumFlatTensorAdapterTest
{
    public class MatrixTests
    {
        private static readonly (int Rows, int Cols)[] NonSquareSizes =
        [
            (1, 3),
            (3, 1),
            (2, 4),
            (4, 2),
        ];

        [TestCaseSource(nameof(NonSquareSizes))]
        public void AddWorksForNonSquareMatrices((int Rows, int Cols) size)
        {
            var (rows, cols) = size;

            var a = new Mat<double>(rows, cols);
            var b = new Mat<double>(rows, cols);
            var c = new Mat<double>(rows, cols);

            for (var row = 0; row < rows; row++)
            {
                for (var col = 0; col < cols; col++)
                {
                    a[row, col] = 100 * row + col + 1;
                    b[row, col] = -10 * row + 2 * col + 0.5;
                }
            }

            Tensor.Add(a.AsTensorSpan(), b.AsTensorSpan(), c.AsTensorSpan());

            for (var row = 0; row < rows; row++)
            {
                for (var col = 0; col < cols; col++)
                {
                    var expected = a[row, col] + b[row, col];
                    Assert.That(c[row, col], Is.EqualTo(expected));
                }
            }
        }


        [Test]
        public void AsTensorSpanDoesNotThrowForSingletonDimensions()
        {
            var rowVector = new Mat<double>(1, 3);
            var colVector = new Mat<double>(3, 1);

            Assert.DoesNotThrow(() => rowVector.AsTensorSpan());
            Assert.DoesNotThrow(() => colVector.AsTensorSpan());
        }

        [Test]
        public void AddWorksForSubMatrixAndDoesNotCorruptOuterArea()
        {
            Mat<double> sourceA =
            [
                [11, 12, 13, 14, 15],
                [21, 22, 23, 24, 25],
                [31, 32, 33, 34, 35],
                [41, 42, 43, 44, 45],
            ];

            Mat<double> sourceB =
            [
                [ 1,  2,  3,  4,  5],
                [ 6,  7,  8,  9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ];

            var destination = new Mat<double>(4, 5);
            destination.Fill(-999);

            var subA = sourceA[1..4, 1..4];
            var subB = sourceB[1..4, 1..4];
            var subC = destination[1..4, 1..4];

            Tensor.Add(subA.AsTensorSpan(), subB.AsTensorSpan(), subC.AsTensorSpan());

            // Verify result in the targeted submatrix.
            for (var row = 0; row < subC.RowCount; row++)
            {
                for (var col = 0; col < subC.ColCount; col++)
                {
                    var expected = subA[row, col] + subB[row, col];
                    Assert.That(subC[row, col], Is.EqualTo(expected));
                }
            }

            // Verify the original areas outside subC are untouched.
            for (var row = 0; row < destination.RowCount; row++)
            {
                for (var col = 0; col < destination.ColCount; col++)
                {
                    var isInsideSub = row is >= 1 and < 4 && col is >= 1 and < 4;
                    if (!isInsideSub)
                    {
                        Assert.That(destination[row, col], Is.EqualTo(-999));
                    }
                }
            }
        }
    }
}
