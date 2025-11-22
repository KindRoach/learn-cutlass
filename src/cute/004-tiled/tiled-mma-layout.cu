#include <string>
#include <cute/atom/mma_atom.hpp>
#include "cute/utils.cuh"

template <typename T, size_t m, size_t n, size_t k, typename Stride>
void mma_example()
{
    using namespace cute;
    TiledMMA mma = make_tiled_mma(
        UniversalFMA<T>{},
        make_layout(make_shape(Int<m>{}, Int<n>{}), Stride{})
    );

    std::string file_name =
        "mma_" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k) +
        (std::is_same<Stride, LayoutLeft>::value ? "_L" : "_R") + ".tex";
    print_latex_to_file(file_name, mma);
}

int main()
{
    using namespace cute;
    mma_example<float, 8, 6, 1, LayoutLeft>();
    mma_example<float, 8, 6, 1, LayoutRight>();
    mma_example<float, 8, 6, 4, LayoutLeft>();
    mma_example<float, 8, 6, 4, LayoutRight>();
}
