#include <iostream>

#include <cute/int_tuple.hpp>

using namespace cute;

int main()
{
    // create tuples
    auto t0 = 1;
    auto t1 = make_tuple(int{1});
    auto t2 = make_tuple(int{2}, Int<3>{});
    auto t3 = make_tuple(int{4}, make_tuple(Int<5>{}, int32_t{6}), Int<7>{});

    // get item
    auto e1 = get<0>(t2); // dynamic 2
    auto e2 = get<1>(t2); // static 3
    std::cout << "t2 elements: " << e1 << ", " << e2.value << "\n";
    std::cout << "\n";

    std::cout << "tuple: " << t0 << "\n";
    std::cout << "  rank : " << rank(t0) << "\n";
    std::cout << "  depth : " << depth(t0) << "\n";
    std::cout << "  size : " << size(t0) << "\n";
    std::cout << "\n";

    std::cout << "tuple: " << t1 << "\n";
    std::cout << "  rank : " << rank(t1) << "\n";
    std::cout << "  depth : " << depth(t1) << "\n";
    std::cout << "  size : " << size(t1) << "\n";
    std::cout << "\n";

    std::cout << "tuple: " << t2 << "\n";
    std::cout << "  rank : " << rank(t2) << "\n";
    std::cout << "  depth : " << depth(t2) << "\n";
    std::cout << "  size : " << size(t2) << "\n";
    std::cout << "\n";

    std::cout << "tuple: " << t3 << "\n";
    std::cout << "  rank : " << rank(t3) << "\n";
    std::cout << "  depth : " << depth(t3) << "\n";
    std::cout << "  size : " << size(t3) << "\n";
    std::cout << "\n";
}
