using System.Numerics.Tensors;
using NumFlat;

namespace NumFlatTensorAdapterTest
{
    public class VectorTests
    {
        private static readonly int[] VectorSizes =
        [
            1,
            3,
            5,
        ];

        [TestCaseSource(nameof(VectorSizes))]
        public void AddWorksForVectors(int count)
        {
            var a = new Vec<double>(count);
            var b = new Vec<double>(count);
            var c = new Vec<double>(count);

            for (var i = 0; i < count; i++)
            {
                a[i] = i + 1;
                b[i] = -3 * i + 0.25;
            }

            Tensor.Add(a.AsTensorSpan(), b.AsTensorSpan(), c.AsTensorSpan());

            for (var i = 0; i < count; i++)
            {
                var expected = a[i] + b[i];
                Assert.That(c[i], Is.EqualTo(expected));
            }
        }

        [Test]
        public void AsTensorSpanDoesNotThrowForSingletonLength()
        {
            var vector = new Vec<double>(1);

            Assert.DoesNotThrow(() => vector.AsTensorSpan());
        }


        [Test]
        public void AddWorksForRowVectorViewsWithStrideGreaterThanOne()
        {
            Mat<double> sourceA =
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34],
            ];

            Mat<double> sourceB =
            [
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
            ];

            var destination = new Mat<double>(3, 4);
            destination.Fill(-999);

            var rowA = sourceA.Rows[1];
            var rowB = sourceB.Rows[1];
            var rowC = destination.Rows[1];

            Assert.That(rowA.Stride, Is.GreaterThan(1));
            Assert.That(rowB.Stride, Is.GreaterThan(1));
            Assert.That(rowC.Stride, Is.GreaterThan(1));

            Tensor.Add(rowA.AsTensorSpan(), rowB.AsTensorSpan(), rowC.AsTensorSpan());

            for (var i = 0; i < rowC.Count; i++)
            {
                var expected = rowA[i] + rowB[i];
                Assert.That(rowC[i], Is.EqualTo(expected));
            }

            for (var row = 0; row < destination.RowCount; row++)
            {
                for (var col = 0; col < destination.ColCount; col++)
                {
                    if (row != 1)
                    {
                        Assert.That(destination[row, col], Is.EqualTo(-999));
                    }
                }
            }
        }

        [Test]
        public void AddWorksForSubvectorAndDoesNotCorruptOuterArea()
        {
            Vec<double> sourceA = [11, 12, 13, 14, 15, 16, 17];
            Vec<double> sourceB = [1, 2, 3, 4, 5, 6, 7];

            var destination = new Vec<double>(7);
            destination.Fill(-999);

            var subA = sourceA[2..6];
            var subB = sourceB[2..6];
            var subC = destination[2..6];

            Tensor.Add(subA.AsTensorSpan(), subB.AsTensorSpan(), subC.AsTensorSpan());

            for (var i = 0; i < subC.Count; i++)
            {
                var expected = subA[i] + subB[i];
                Assert.That(subC[i], Is.EqualTo(expected));
            }

            for (var i = 0; i < destination.Count; i++)
            {
                var isInsideSub = i is >= 2 and < 6;
                if (!isInsideSub)
                {
                    Assert.That(destination[i], Is.EqualTo(-999));
                }
            }
        }
    }
}
