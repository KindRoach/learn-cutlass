#include <iostream>

#include <cute/layout.hpp>

int main()
{
    using namespace cute;

    // Define a nested Shape: (4, (3, 6))
    // Equivalent to a 2-level Layout: first dimension 4, second dimension is a tuple (3,6)
    Layout<Shape<_4, Shape<_3, _6>>> a{};

    std::cout << "Layout a: " << a << "\n";

    // Operations based on IntTuple
    std::cout << "rank(a)   = " << rank(a) << "\n";
    std::cout << "depth(a)  = " << depth(a) << "\n";
    std::cout << "shape(a)  = " << shape(a) << "\n";
    std::cout << "stride(a) = " << stride(a) << "\n";
    std::cout << "size(a)   = " << size(a) << "\n";
    std::cout << "cosize(a) = " << cosize(a) << "\n";

    // Access sub-layouts
    auto a0 = get<0>(a);
    auto a1 = get<1>(a);

    std::cout << "\nSub-layouts:\n";
    std::cout << "get<0>(a): " << a0 << ", shape: " << shape(a0) << ", stride: " << stride(a0) << "\n";
    std::cout << "get<1>(a): " << a1 << ", shape: " << shape(a1) << ", stride: " << stride(a1) << "\n";

    // Simplified nested access operations
    std::cout << "\nNested access:\n";
    std::cout << "rank<1>(a)   = " << rank<1>(a) << "\n";
    std::cout << "depth<1>(a)  = " << depth<1>(a) << "\n";
    std::cout << "shape<1>(a)  = " << shape<1>(a) << "\n";
    std::cout << "size<1>(a)   = " << size<1>(a) << "\n";

    // Deeper nested access: first element of a1
    std::cout << "shape<1,0>(a) = " << shape<1, 0>(a) << "\n";
    std::cout << "size<1,0>(a)  = " << size<1, 0>(a) << "\n";
}
