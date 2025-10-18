#include <cute/tensor.hpp>


template <class Tensor, class Tiler>
void inner_partition_with_print(Tensor A, Tiler tiler)
{
    using namespace cute;

    std::cout << "=============== inner partition ===============\n";

    auto rest_shape = shape<1>(zipped_divide(layout(A), tiler));
    std::cout << "tiler = " << tiler << "\n";
    std::cout << "tile_layout = " << rest_shape << "\n";
    std::cout << "\n";

    std::cout << "A = ";
    print_tensor(A);
    std::cout << "\n";

    for (int i = 0; i < size(rest_shape); ++i)
    {
        auto coord = idx2crd(i, rest_shape);

        // identical to local_tile(A , tiler, coord)
        auto inner_a = inner_partition(A, tiler, coord);

        std::cout << "inner_partition(A, tiler, " << coord << ") = ";
        print_tensor(inner_a);
        std::cout << "\n";
    }
}

template <class Tensor, class Tiler>
void outer_partition_with_print(Tensor A, Tiler tiler)
{
    using namespace cute;

    std::cout << "=============== outer partition ===============\n";

    std::cout << "tiler = " << tiler << "\n";
    std::cout << "\n";

    std::cout << "A = ";
    print_tensor(A);
    std::cout << "\n";

    for (int i = 0; i < size(tiler); ++i)
    {
        auto coord = idx2crd(i, tiler);

        // identical to local_tile(A , tiler, coord)
        auto outer_a = outer_partition(A, tiler, coord);

        std::cout << "outer_partition(A, tiler, " << coord << ") = ";
        print_tensor(outer_a);
        std::cout << "\n";
    }
}

template <class Tensor, class Layout>
void tv_partition_with_print(Tensor A, Layout tv_layout)
{
    using namespace cute;

    std::cout << "=============== tv partition ===============\n";

    std::cout << "tv_layout = " << tv_layout << "\n";
    std::cout << "\n";

    std::cout << "A = ";
    print_tensor(A);
    std::cout << "\n";

    auto tv = composition(A, tv_layout);
    std::cout << "tv = A o tv_layout = ";
    print_tensor(tv);
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

    auto tiler = Shape<_2, _4>{};

    inner_partition_with_print(A, tiler);
    outer_partition_with_print(A, tiler);

    auto tv_layout = Layout<Shape<Shape<_2, _4>, Shape<_2, _2>>,
                            Stride<Stride<_8, _1>, Stride<_4, _16>>>{};

    tv_partition_with_print(A, tv_layout);
}
