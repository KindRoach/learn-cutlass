#include <iostream>

#include <cute/layout.hpp>

int main()
{
    using namespace cute;

    // single dimension Layout
    Layout s8 = make_layout(Int<8>{});
    Layout d8 = make_layout(8);

    // multi dimension Layout
    Layout s2xs4 = make_layout(make_shape(Int<2>{}, Int<4>{}));
    Layout s2xd4 = make_layout(make_shape(Int<2>{}, 4));

    // with custom stride
    Layout s2xd4_a = make_layout(make_shape(Int<2>{}, 4),
                                 make_stride(Int<12>{}, Int<1>{}));

    // with auto stride
    Layout s2xd4_col = make_layout(make_shape(Int<2>{}, 4), LayoutLeft{});
    Layout s2xd4_row = make_layout(make_shape(Int<2>{}, 4), LayoutRight{});

    // With exist Layout
    Layout s2xh4 = make_layout(make_shape(2, make_shape(2, 2)),
                               make_stride(4, make_stride(2, 1)));
    Layout s2xh4_col = make_layout(shape(s2xh4), LayoutLeft{});

    std::cout << "s8 = " << s8 << "\n";
    std::cout << "d8 = " << d8 << "\n";
    std::cout << "s2xs4 = " << s2xs4 << "\n";
    std::cout << "s2xd4 = " << s2xd4 << "\n";
    std::cout << "s2xd4_a = " << s2xd4_a << "\n";
    std::cout << "s2xd4_col = " << s2xd4_col << "\n";
    std::cout << "s2xd4_row = " << s2xd4_row << "\n";
    std::cout << "s2xh4 = " << s2xh4 << "\n";
    std::cout << "s2xh4_col = " << s2xh4_col << "\n";
}
