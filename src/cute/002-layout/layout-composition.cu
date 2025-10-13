#include <iostream>

#include <cute/tensor.hpp>

template <class LShape, class LStride,
          class RShape, class RStride>
void composition_witch_print(
    cute::Layout<LShape, LStride> const& A,
    cute::Layout<RShape, RStride> const& B)
{
    using namespace cute;

    auto C = composition(A, B);

    print_layout(A);
    print("\n");

    print_layout(B);
    print("\n");

    print_layout(C);
    print("\n");

    for (int i = 0; i < size(C); ++i)
    {
        std::cout << "C(" << i << ") = "
            << "A(B(" << i << ")) = "
            << "A(B(" << idx2crd(i, shape(B)) << ")) = "
            << "A(" << B(i) << ") = "
            << "A(" << idx2crd(B(i), shape(A)) << ") = "
            << C(i) << "\n";
    }
}


int main()
{
    using namespace cute;

    composition_witch_print(
        make_layout(make_shape(6, 2), make_shape(8, 2)),
        make_layout(make_shape(4, 3), make_shape(3, 1))
    );
}
