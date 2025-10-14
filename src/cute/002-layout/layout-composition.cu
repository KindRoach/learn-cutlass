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

    print("A = ");
    print_layout(A);
    print("\n");

    print("B = ");
    print_layout(B);
    print("\n");

    print("C = A o B = ");
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

    std::cout << "\n";
}

void by_mode_composition_example()
{
    using namespace cute;

    // (12,(4,8)):(59,(13,1))
    auto A = make_layout(
        make_shape(12, make_shape(4, 8)),
        make_stride(59, make_stride(13, 1))
    );

    std::cout << "A = " << A << "\n";

    // <3:4, 8:2>
    auto tiler = make_tile(
        Layout<_3, _4>{}, // Apply 3:4 to mode-0
        Layout<_8, _2>{} // Apply 8:2 to mode-1
    );

    std::cout << "tiler = " << tiler << "\n";

    // (_3,(2,4)):(236,(26,1))
    auto result = composition(A, tiler);

    print("result = A o tiler = ");
    print_layout(result);
    print("\n");

    // Identical to
    auto same_r = make_layout(
        composition(layout<0>(A), get<0>(tiler)),
        composition(layout<1>(A), get<1>(tiler))
    );

    print("same_r = ");
    print_layout(same_r);
    print("\n");
}


int main()
{
    using namespace cute;

    composition_witch_print(
        make_layout(make_shape(6, 2), make_shape(8, 2)),
        make_layout(make_shape(4, 3), make_shape(3, 1))
    );

    composition_witch_print(
        make_layout(make_shape(10, 2), make_shape(16, 4)),
        make_layout(make_shape(5, 4), make_shape(1, 5))
    );

    by_mode_composition_example();
}
