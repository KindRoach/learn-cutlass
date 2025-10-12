#include <iostream>

#include <cute/tensor.hpp>

int main()
{
    using namespace cute;

    auto a = Layout<
        Shape<_2, Shape<_1, _6>>,
        Stride<_1, Stride<_6, _2>>
    >{};
    std::cout << "a = " << a << "\n";

    auto b = coalesce(a);
    std::cout << "coalesce(a) = " << b << "\n";

    auto c = coalesce(a, Step<_1, _1>{});
    std::cout << "coalesce(a, Step<_1,_1>) = " << c << "\n";

    auto d = make_layout(coalesce(layout<0>(a)), coalesce(layout<1>(a)));
    std::cout << "make_layout(coalesce(layout<0>(a)), coalesce(layout<1>(a))) = " << d << "\n\n";

    auto e = Layout<
        Shape<Shape<_2, _1>, Shape<_3, _4>>,
        Stride<Stride<_1, _1>, Stride<_4, _2>>
    >{};
    std::cout << "e = " << e << "\n";

    auto f = coalesce(e);
    std::cout << "coalesce(e) = " << f << "\n";

    auto g = coalesce(e, Step<_1, Step<_1, _1>>{});
    std::cout << "coalesce(e, Step<_1, Step<_1,_1>>) = " << g << "\n";
}
