#include <iostream>

#include <cute/int_tuple.hpp>

using namespace cute;

int main()
{
    // create tuples
    auto t1 = make_tuple(int{2}, Int<3>{});
    auto t2 = make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{});

    // rank
    std::cout << "t1 rank: " << rank(t1) << "\n";
    std::cout << "t2 rank: " << rank(t2) << "\n";

    // depth
    std::cout << "t1 depth: " << depth(t1) << "\n";
    std::cout << "t2 depth: " << depth(t2) << "\n";

    // get
    auto e1 = get<0>(t1); // dynamic 2
    auto e2 = get<1>(t1); // static 3
    std::cout << "t1 elements: " << e1 << ", " << e2.value << "\n";

    // size
    std::cout << "t1 size: " << size(t1) << "\n";
    std::cout << "t2 size: " << size(t2) << "\n";
}
