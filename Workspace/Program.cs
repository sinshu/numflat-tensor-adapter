using System;
using System.Numerics.Tensors;
using NumFlat;


public static class Program
{
    static void Main(string[] args)
    {
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

        Tensor.Add(a.AsTensorSpan(), b.AsTensorSpan(), c.AsTensorSpan());

        Console.WriteLine(c);
    }
}
