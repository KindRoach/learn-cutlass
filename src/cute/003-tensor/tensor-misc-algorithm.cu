#include <iostream>

#include <cute/tensor.hpp>

template <class Tensor>
void clear_tensor(Tensor& A)
{
    using namespace cute;

    auto B = A;
    clear(B);

    std::cout << "clear(A) = ";
    print_tensor(B);
    std::cout << "\n";
}

template <class Tensor>
void fill_tensor(Tensor& A)
{
    using namespace cute;

    auto B = A;
    fill(B, 42);

    std::cout << "fill(A, 42) = ";
    print_tensor(B);
    std::cout << "\n";
}

template <class Tensor>
void transform_tensor(Tensor& A)
{
    using namespace cute;

    auto B = A;
    transform(B,
              [](auto x)
              {
                  return x * x;
              }
    );

    std::cout << "transform x^2 = ";
    print_tensor(B);
    std::cout << "\n";
}

template <class Tensor>
void foreach_tensor(Tensor& A)
{
    using namespace cute;

    auto B = A;
    for_each(B,
             [](auto& x)
             {
                 x = x * x;
             }
    );

    std::cout << "foreach x^2 = ";
    print_tensor(B);
    std::cout << "\n";
}

template <class Tensor>
void copy_tensor(Tensor& A)
{
    using namespace cute;

    auto B = make_tensor_like(A);
    copy(A, B);

    std::cout << " Copy(A) = ";
    print_tensor(B);
    std::cout << "\n";
}

template <class Tensor>
void copy_tensor_on_odd(Tensor& A)
{
    using namespace cute;

    auto pred = make_tensor_like(A);
    for (int i = 0; i < size(pred); ++i)
    {
        // Predicate: true for odd elements
        pred[i] = A[i] % 2 != 0;
    }

    auto B = make_tensor_like(A);
    copy_if(pred, A, B);

    std::cout << "Pred Tensor = ";
    print_tensor(pred);
    std::cout << "\n";

    std::cout << "Copy_if(A) on odd elements = ";
    print_tensor(B);
    std::cout << "\n";
}


int main()
{
    using namespace cute;

    auto A = make_tensor<int>(Shape<_4, _8>{});
    for (int i = 0; i < size(A); ++i)
    {
        A[i] = i;
    }

    std::cout << "Source Tensor = ";
    print_tensor(A);
    std::cout << "\n";

    clear_tensor(A);
    fill_tensor(A);
    transform_tensor(A);
    foreach_tensor(A);
    copy_tensor(A);
    copy_tensor_on_odd(A);
}
