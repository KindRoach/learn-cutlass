#include <iostream>

#include <cute/layout.hpp>

void sub_layout_example()
{
    using namespace cute;

    Layout a = Layout<Shape<_4, Shape<_3, _6>>>{};
    Layout a0 = layout<0>(a);
    Layout a1 = layout<1>(a);
    Layout a10 = layout<1, 0>(a);
    Layout a11 = layout<1, 1>(a);

    std::cout << "a              = " << a << "\n";
    std::cout << "layout<0>(a)   = " << a0 << "\n";
    std::cout << "layout<1>(a)   = " << a1 << "\n";
    std::cout << "layout<1,0>(a) = " << a10 << "\n";
    std::cout << "layout<1,1>(a) = " << a11 << "\n";
    std::cout << "\n";
}

void select_example()
{
    using namespace cute;

    Layout a = Layout<Shape<_2, _3, _5, _7>>{};
    Layout a13 = select<1, 3>(a);
    Layout a01 = select<0, 1, 3>(a);
    Layout a2 = select<2>(a);

    std::cout << "a                = " << a << "\n";
    std::cout << "select<1,3>(a)   = " << a13 << "\n";
    std::cout << "select<0,1,3>(a) = " << a01 << "\n";
    std::cout << "select<2>(a)     = " << a2 << "\n";
    std::cout << "\n";
}

void take_example()
{
    using namespace cute;

    Layout a = Layout<Shape<_2, _3, _5, _7>>{};
    Layout a13 = take<1, 3>(a);
    Layout a14 = take<1, 4>(a);
    // take<1,1> not allowed. Empty layouts not allowed.

    std::cout << "a            = " << a << "\n";
    std::cout << "take<1,3>(a) = " << a13 << "\n";
    std::cout << "take<1,4>(a) = " << a14 << "\n";
    std::cout << "\n";
}

void concatenation_example()
{
    using namespace cute;

    Layout a = Layout<_3, _1>{};
    Layout b = Layout<_4, _3>{};
    Layout ab = make_layout(a, b);
    Layout ba = make_layout(b, a);
    Layout ab_ba = make_layout(make_layout(a, b), make_layout(b, a));
    Layout aa = make_layout(a);
    Layout aaa = make_layout(aa);
    Layout a_a_a = make_layout(a, make_layout(a), a);

    std::cout << "a                                 = " << a << "\n";
    std::cout << "b                                 = " << b << "\n";
    std::cout << "ab = make_layout(a, b)            = " << ab << "\n";
    std::cout << "ba = make_layout(b, a)            = " << ba << "\n";
    std::cout << "make_layout(ab, ba)               = " << ab_ba << "\n";
    std::cout << "aa = make_layout(a)               = " << aa << "\n";
    std::cout << "make_layout(aa)                   = " << aaa << "\n";
    std::cout << "make_layout(a, make_layout(a), a) = " << a_a_a << "\n";
    std::cout << "\n";
}

void append_prepend_replace_example()
{
    using namespace cute;

    Layout a = Layout<_3, _1>{};
    Layout b = Layout<_4, _3>{};
    Layout ab = append(a, b);
    Layout ba = prepend(a, b);
    Layout c  = append(ab, ab);
    Layout d  = replace<2>(c, b);

    std::cout << "a                 = " << a << "\n";
    std::cout << "b                 = " << b << "\n";
    std::cout << "ab = append(a,b)  = " << ab << "\n";
    std::cout << "ba = prepend(a,b) = " << ba << "\n";
    std::cout << "c = append(ab,ab) = " << c << "\n";
    std::cout << "replace<2>(c,b)   = " << d << "\n";
    std::cout << "\n";
}

void group_flatten_example()
{
    using namespace cute;

    Layout a = Layout<Shape<_2, _3, _5, _7>>{};
    Layout b = group<0, 2>(a);
    Layout c = group<1, 3>(b);
    Layout e = flatten(a);
    Layout f = flatten(b);
    Layout g = flatten(c);

    std::cout << "a                 = " << a << "\n";
    std::cout << "b = group<0,2>(a) = " << b << "\n";
    std::cout << "c = group<1,3>(b) = " << c << "\n";
    std::cout << "flatten(a)        = " << e << "\n";
    std::cout << "flatten(b)        = " << f << "\n";
    std::cout << "flatten(c)        = " << g << "\n";
    std::cout << "\n";
}

int main()
{
    using namespace cute;
    sub_layout_example();
    select_example();
    take_example();
    concatenation_example();
    append_prepend_replace_example();
    group_flatten_example();
}
