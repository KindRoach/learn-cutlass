#include <iostream>

#include <cute/tensor.hpp>

int main()
{
    using namespace cute;

    auto x = make_tensor<int>(Shape<_4, _8>{});
    auto y = make_tensor<int>(Shape<_4, _8>{});

    for (int i = 0; i < size(x); ++i)
    {
        x[i] = i;
        y[i] = size(x) - i;
    }

    std::cout << "Tensor x = ";
    print_tensor(x);
    std::cout << "\n";

    std::cout << "Tensor y = ";
    print_tensor(y);
    std::cout << "\n";

    int alpha = 2;
    int beta  = 3;

    // y = alpha * x + beta * y
    axpby(alpha, x, beta, y);

    std::cout << "y = " << alpha << " * x + " << beta << " * y = ";
    print_tensor(y);
    std::cout << "\n";
}
