#include <iostream>

#include <cute/layout.hpp>

int main()
{
    using namespace cute;

    // layout = (3, (2, 3))
    auto layout = make_layout(Shape<_3, Shape<_2, _3>>{});
    std::cout << layout << "\n";

    auto shape = layout.shape();
    std::cout << idx2crd(16, shape) << "\n";
    std::cout << idx2crd(make_coord(1, _6{}), shape) << "\n";
    std::cout << idx2crd(make_coord(1, make_coord(1, 2)), shape) << "\n";

    auto stride = layout.stride();
    std::cout << crd2idx(16, shape, stride) << "\n";
    std::cout << crd2idx(make_coord(1, 5), shape, stride) << "\n";
    std::cout << crd2idx(make_coord(1, make_coord(1, 2)), shape, stride) << "\n";

    auto stride_custom = Stride<_3, Stride<_12, _1>>{};
    std::cout << crd2idx(16, shape, stride_custom) << "\n";
    std::cout << crd2idx(make_coord(1, 5), shape, stride_custom) << "\n";
    std::cout << crd2idx(make_coord(1, make_coord(1, 2)), shape, stride_custom) << "\n";
}
