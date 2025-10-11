#include <iostream>

#include <cute/tensor.hpp>

using namespace cute;

int main()
{
    // Create a Tensor with shape (2,(3,4))
    // Outer layer has 2 elements, each element is a 3x4 matrix
    Tensor t = make_tensor<int>(Shape<_2, Shape<_3, _4>>{});

    // Fill data
    for (int i = 0; i < size(t); i++)
    {
        t(i) = i;
    }

    print("t = ");
    print_tensor(t);
    print("\n");


    // .data() returns raw pointer to underlying storage
    std::cout << "t.data() = " << t.data() << "\n";
    std::cout << "*t.data() = " << *t.data() << "\n";

    // .size() returns total number of elements
    std::cout << "t.size() = " << t.size() << "\n";

    // Access elements via coordinates
    std::cout << "t[make_coord(1, make_coord(2, 3))] = " << t[make_coord(1, make_coord(2, 3))] << "\n";
    std::cout << "t(make_coord(1, make_coord(2, 3))) = " << t(make_coord(1, make_coord(2, 3))) << "\n";
    std::cout << "t(1, make_coord(2, 3)) = : " << t(1, make_coord(2, 3)) << "\n\n";

    // Hierarchical APIs
    std::cout << "rank<1>(t): " << rank<1>(t) << "\n";
    std::cout << "depth<1>(t): " << depth<1>(t) << "\n";
    std::cout << "shape<1>(t): " << shape<1>(t) << "\n";
    std::cout << "size<1>(t): " << size<1>(t) << "\n";
    std::cout << "layout<1>(t): " << layout<1>(t) << "\n\n";

    // Extract sub-tensor
    Tensor sub0 = tensor<0>(t);
    print("tensor<0>(t) = ");
    print(sub0);
    print("\n");

    Tensor sub1 = tensor<1>(t);
    print("tensor<1>(t) = ");
    print(sub1);
    print("\n");
}
