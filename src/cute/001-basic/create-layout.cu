#include <cute/layout.hpp>

template <class Shape, class Stride>
void print_layout_with_name(const cute::Layout<Shape, Stride>& layout, const std::string& name)
{
    using namespace cute;
    print((name + " = ").c_str());
    print(layout);
    print("\n");
}

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

    print_layout_with_name(s8, "s8");
    print_layout_with_name(d8, "d8");
    print_layout_with_name(s2xs4, "s2xs4");
    print_layout_with_name(s2xd4, "s2xd4");
    print_layout_with_name(s2xd4_a, "s2xd4_a");
    print_layout_with_name(s2xd4_col, "s2xd4_col");
    print_layout_with_name(s2xd4_row, "s2xd4_row");
    print_layout_with_name(s2xh4, "s2xh4");
    print_layout_with_name(s2xh4_col, "s2xh4_col");
}
