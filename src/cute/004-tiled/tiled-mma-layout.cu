#include <string>
#include <cute/atom/mma_atom.hpp>
#include "cute/utils.cuh"

template <typename T, size_t m, size_t n, size_t k, typename Stride>
void universal_fma_example()
{
    using namespace cute;
    TiledMMA mma = make_tiled_mma(
        UniversalFMA<T>{},
        make_layout(make_shape(Int<m>{}, Int<n>{}), Stride{})
    );

    std::string name =
        "mma_" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k) +
        (std::is_same<Stride, LayoutLeft>::value ? "_L" : "_R");

    std::cout << "\n" << name << "\n" << "size: " << size(mma) << " threads" << "\n";
    print(mma);
    print_latex_to_file(name + ".tex", mma);
}

template <size_t m, size_t n, size_t k, size_t atom_m, size_t atom_n, size_t atom_k>
void tensor_core_mma_example()
{
    using namespace cute;
    TiledMMA mma = make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{},
        make_layout(make_shape(Int<atom_m>{}, Int<atom_n>{}, Int<atom_k>{})), // MMA Atoms Layout
        make_tile(Int<m>{}, Int<n>{}, Int<k>{}) // MMA Tile
    );

    std::string name = "SM80_16x8x16_F16F16F16F16_TN_" +
        std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k) + "_" +
        std::to_string(atom_m) + "x" + std::to_string(atom_n) + "x" + std::to_string(atom_k);

    std::cout << "\n" << name << "\n" << "size: " << size(mma) << " threads" << "\n";
    print(mma);
    print_latex_to_file(name + ".tex", mma);
}

int main()
{
    using namespace cute;
    universal_fma_example<float, 8, 6, 1, LayoutLeft>();
    universal_fma_example<float, 8, 6, 1, LayoutRight>();
    universal_fma_example<float, 8, 6, 4, LayoutLeft>();
    universal_fma_example<float, 8, 6, 4, LayoutRight>();
    tensor_core_mma_example<32, 32, 16, 2, 2, 1>();
    tensor_core_mma_example<64, 64, 16, 2, 2, 1>();
    tensor_core_mma_example<64, 64, 16, 4, 4, 1>();
}
