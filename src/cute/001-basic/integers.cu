#include <iostream>

#include <cute/numeric/integral_constant.hpp>

using namespace cute;

int main()
{
    // === Dynamic integers ===
    int a = 5;
    size_t b = 10;

    std::cout << "Dynamic integers: a = " << a << ", b = " << b << std::endl;

    // === Static integers ===
    auto c = C<3>{};
    auto d = C<7>{};

    std::cout << "Static integers: c = " << int(c) << ", d = " << int(d) << std::endl;

    // === Arithmetic operations ===
    auto e = c + d; // type(e) = C<10>
    std::cout << "Static addition result (c + d) = " << int(e)
        << " (type is still static: " << std::boolalpha
        << is_static<decltype(e)>::value << ")" << std::endl;

    // Mixed operation: static + dynamic
    auto f = c + a;
    std::cout << "Mixed addition (c + a) = " << f << " (runtime value)" << std::endl;

    // === Type traits ===
    std::cout << "is_integral<int>: " << is_integral<int>::value << std::endl;
    std::cout << "is_integral<C<3>>: " << is_integral<C<3>>::value << std::endl;
    std::cout << "is_constant<3, C<3>>: " << is_constant<3, C<3>>::value << std::endl;
}
