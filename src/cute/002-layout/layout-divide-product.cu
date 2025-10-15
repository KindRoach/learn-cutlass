#include <iostream>

#include <cute/layout.hpp>

template <class Shape, class Stride>
void divide_witch_print(cute::Layout<Shape, Stride> const& A, auto B)
{
    using namespace cute;

    std::cout << "A = " << A << "\n";
    std::cout << "B = " << B << "\n";
    std::cout << "logical_divide(A, B) = " << logical_divide(A, B) << "\n";
    std::cout << "zipped_divide(A, B) = " << zipped_divide(A, B) << "\n";
    std::cout << "tiled_divide(A, B) = " << tiled_divide(A, B) << "\n";
    std::cout << "flat_divide(A, B) = " << flat_divide(A, B) << "\n\n";
}

template <class Shape, class Stride>
void product_witch_print(cute::Layout<Shape, Stride> const& A, auto B)
{
    using namespace cute;

    std::cout << "A = " << A << "\n";
    std::cout << "B = " << B << "\n";
    std::cout << "logical_product(A, B) = " << logical_product(A, B) << "\n";
    std::cout << "zipped_product(A, B) = " << zipped_product(A, B) << "\n";
    std::cout << "tiled_product(A, B) = " << tiled_product(A, B) << "\n";
    std::cout << "flat_product(A, B) = " << flat_product(A, B) << "\n\n";
}


int main()
{
    using namespace cute;

    divide_witch_print(
        Layout<Shape<_4, _2, _3>, Shape<_2, _1, _8>>{},
        Layout<_4, _2>{}
    );

    divide_witch_print(
        Layout<Shape<_9, Shape<_4, _8>>, Shape<Int<59>, Shape<Int<13>, _1>>>{},
        Tile<Layout<_3, _3>, Layout<Shape<_2, _4>, Shape<_1, _8>>>{}
    );

    product_witch_print(
        Layout<Shape<_2, _2>, Shape<_4, _1>>{},
        Layout<_6, _1>{}
    );

    product_witch_print(
        Layout<Shape<_2, _5>, Shape<_5, _1>>{},
        Tile<Layout<_3, _5>, Layout<_4, _6>>{}
    );
}
