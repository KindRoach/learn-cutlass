#include <iostream>

#include <cute/tensor.hpp>


template <class Shape, class Stride>
void complement_witch_print(cute::Layout<Shape, Stride> const& A, auto size)
{
    using namespace cute;

    auto B = complement(A, size);

    std::cout << "A = " << A << "\n";
    std::cout << "complement(A, " << size << ") = " << B << "\n\n";
}

int main()
{
    using namespace cute;

    complement_witch_print(Layout<_4, _1>{}, Int<24>{});
    complement_witch_print(Layout<_6, _4>{}, Int<24>{});
    complement_witch_print(Layout<_4, _2>{}, Int<24>{});
    complement_witch_print(Layout<Shape<_4, _6>, Shape<_1, _4>>{}, Int<24>{});
    complement_witch_print(Layout<Shape<_2, _4>, Shape<_1, _6>>{}, Int<24>{});
    complement_witch_print(Layout<Shape<_2, _2>, Shape<_1, _6>>{}, Int<24>{});
}
