#include <cute/tensor.hpp>

template <typename T>
void access_single_item()
{
    using namespace cute;

    Tensor A = make_tensor<T>(Shape<Shape<_4, _5>, Int<13>>{}, Stride<Stride<_12, _1>, _64>{});
    Tensor B = make_tensor<T>(Shape<Int<13>, Int<20>>{});

    // Fill A via natural coordinates op[]
    for (int m0 = 0; m0 < size<0, 0>(A); ++m0)
        for (int m1 = 0; m1 < size<0, 1>(A); ++m1)
            for (int n = 0; n < size<1>(A); ++n)
                A[make_coord(make_coord(m0, m1), n)] = n + 2 * m0;

    print("A = ");
    print_tensor(A);
    print("\n");

    // Transpose A into B using variadic op()
    for (int m = 0; m < size<0>(A); ++m)
        for (int n = 0; n < size<1>(A); ++n)
            B(n, m) = A(m, n);

    print("B = ");
    print_tensor(B);
    print("\n");

    // Copy B to A as if they are arrays
    for (int i = 0; i < A.size(); ++i)
        A[i] = B[i];

    print("A = ");
    print_tensor(A);
    print("\n");
}

template <typename T>
void slice_tensor()
{
    using namespace cute;

    // ((_3,2),(2,_5,_2)):((4,1),(_2,13,100))
    Tensor A = make_tensor<T>(Shape<Shape<_3, _2>, Shape<_2, _5, _2>>{},
                              Stride<Stride<_4, _1>, Stride<_2, Int<13>, Int<100>>>{});

    for (int i = 0; i < size(A); ++i)
    {
        A[i] = i;
    }

    print("A = ");
    print_tensor(A);
    print("\n");

    // ((2,_5,_2)):((_2,13,100))
    Tensor B = A(2, _);
    print("A(2, _) = ");
    print_tensor(B);
    print("\n");

    // ((_3,_2)):((4,1))
    Tensor C = A(_, 5);
    print("A(_, 5) = ");
    print_tensor(C);
    print("\n");

    // (_3,2):(4,1)
    Tensor D = A(make_coord(_, _), 5);
    print("A(make_coord(_, _), 5) = ");
    print_tensor(D);
    print("\n");

    // (_3,_5):(4,13)
    Tensor E = A(make_coord(_, 1), make_coord(0, _, 1));
    print("A(make_coord(_, 1), make_coord(0, _, 1)) = ");
    print_tensor(E);
    print("\n");

    // (2,2,_2):(1,_2,100)
    Tensor F = A(make_coord(2, _), make_coord(_, 3, _));
    print("A(make_coord(2, _), make_coord(_, 3, _)) = ");
    print_tensor(F);
    print("\n");
}

int main()
{
    using namespace cute;
    access_single_item<int>();
    slice_tensor<int>();
}
