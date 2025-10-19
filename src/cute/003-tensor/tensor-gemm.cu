#include <iostream>

#include <cute/tensor.hpp>

void gemm_example_matrix_multiply()
{
    using namespace cute;
    auto A = make_tensor<int>(Shape<_4, _8>{});
    auto B = make_tensor<int>(Shape<_5, _8>{});
    auto C = make_tensor<int>(Shape<_4, _5>{});

    for (int i = 0; i < size(A); ++i)
    {
        A[i] = i;
    }

    for (int i = 0; i < size(B); ++i)
    {
        B[i] = size(B) - i;
    }

    fill(C, 1);

    std::cout << "Tensor A = ";
    print_tensor(A);
    std::cout << "\n";

    std::cout << "Tensor B = ";
    print_tensor(B);
    std::cout << "\n";

    std::cout << "Tensor C = ";
    print_tensor(C);
    std::cout << "\n";

    gemm(A, B, C);

    std::cout << "C += A x B = ";
    print_tensor(C);
    std::cout << "\n";
}

int main()
{
    gemm_example_matrix_multiply();
}
