#include <iostream>

#include <cute/layout.hpp>

int main()
{
    using namespace cute;

    // layout = (3, (2, 3))
    auto layout = make_layout(Shape<_3,Shape<_2,_3>>{});
    std::cout << "layout = " << layout << "\n";


    std::cout << "\naccess with 1-D coordinate:\n";
    for (int i = 0; i < size(layout); ++i)
    {
        std::cout << "layout(" << i << ") = " << layout(i) << "\n";
    }

    std::cout << "\naccess with 2-D coordinate:\n";
    for (int i = 0; i < size<0>(layout); ++i)
    {
        for (int j = 0; j < size<1>(layout); ++j)
        {
            std::cout << "layout(" << i << ", " << j << ") = " << layout(make_coord(i, j)) << "\n";
        }
    }

    std::cout << "\naccess with 3-D coordinate:\n";
    for (int i = 0; i < size<0>(layout); ++i)
    {
        auto layout_1 = get<1>(layout);
        for (int j = 0; j < size<0>(layout_1); ++j)
        {
            for (int k = 0; k < size<1>(layout_1); ++k)
            {
                std::cout << "layout(" << i << ", (" << j << ", " << k << ")) = " << layout(
                    make_coord(i, make_coord(j, k))) << "\n";
            }
        }
    }
}
