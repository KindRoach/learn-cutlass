#include <cute/tensor.hpp>
#include <thrust/device_vector.h>

#include <cpp-bench-utils/utils.hpp>
#include <cute/utils.cuh>

template <
    class T,
    class AGlobalLayout, class ASmemLayout,
    class BGlobalLayout, class BSmemLayout,
    class CGlobalLayout, class TiledMmaC
>
__global__ static
__launch_bounds__(decltype(size(TiledMmaC{}))::value)
void tiled_mma_kernel(
    T const* A, AGlobalLayout mA_layout, ASmemLayout sA_layout,
    T const* B, BGlobalLayout mB_layout, BSmemLayout sB_layout,
    T* C, CGlobalLayout mC_layout, TiledMmaC mma_c)
{
    using namespace cute;

    Tensor mA = make_tensor(make_gmem_ptr(A), mA_layout); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), mB_layout); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), mC_layout); // (M,N)

    // Shared memory buffers
    __shared__ T smemA[cosize_v<ASmemLayout>];
    __shared__ T smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K)

    if (thread(1))
    {
        copy(mA, sA);
        copy(mB, sB);
    }

    __syncthreads();

    ThrMMA thr_mma = mma_c.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(mC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    clear(tCrC);
    gemm(mma_c, tCsA, tCsB, tCrC);
    copy(tCrC, tCgC);

    if (thread(1))
    {
        print_tensor_with_label("mA : ", mA);
        print_tensor_with_label("mB : ", mB);
        print_tensor_with_label("mC : ", mC);

        print_tensor_with_label("sA : ", sA);
        print_tensor_with_label("sB : ", sB);

        print_tensor_with_label("tCsA : ", tCsA);
        print_tensor_with_label("tCsB : ", tCsB);
        print_tensor_with_label("tCgC : ", tCgC);
        print_tensor_with_label("tCrC: ", tCrC);
    }
}

template <size_t m, size_t n, size_t k, typename T, class TiledMMA>
void tiled_mma_example()
{
    using namespace cbu;
    using namespace cute;

    std::vector<T> a(m * k), b(n * k), c(m * n);

    for (int i = 0; i < a.size(); ++i)
    {
        a[i] = static_cast<T>(i * 0.01);
    }

    for (int i = 0; i < b.size(); ++i)
    {
        b[i] = static_cast<T>(i * 0.01);
    }

    matrix_multiply_ref<T, matrix_layout::col_major>(a, b, c, m, n, k);
    thrust::device_vector<T> d_a = a, d_b = b, d_c(m * n);

    Layout mA_layout = make_layout(make_shape(Int<m>{}, Int<k>{}), LayoutRight{});
    Layout mB_layout = make_layout(make_shape(Int<n>{}, Int<k>{}), LayoutRight{});
    Layout mC_layout = make_layout(make_shape(Int<m>{}, Int<n>{}), LayoutRight{});

    Layout sA_layout = make_layout(make_shape(Int<m>{}, Int<k>{}));
    Layout sB_layout = make_layout(make_shape(Int<n>{}, Int<k>{}));

    TiledMMA mma{};

    // launch kernel
    dim3 dimBlock(size(mma));
    dim3 dimGrid(1, 1);
    tiled_mma_kernel<<<dimGrid, dimBlock>>>(
        d_a.data().get(), mA_layout, sA_layout,
        d_b.data().get(), mB_layout, sB_layout,
        d_c.data().get(), mC_layout,
        mma
    );

    cudaDeviceSynchronize();
    cuda_acc_check(c, d_c);
}

template <size_t m, size_t n, size_t k>
void universal_fma_example()
{
    std::cout << "Universal FMA Tiled MMA Example (M=" << m << ", N=" << n << ", K=" << k << ")\n";

    using namespace cute;
    using dtype = float;

    TiledMMA mma = make_tiled_mma(
        UniversalFMA<dtype>{},
        Layout<Shape<Int<m / 2>, Int<n / 2>, _1>>{}
    );

    tiled_mma_example<m, n, k, dtype, decltype(mma)>();
}


template <size_t atom_m, size_t atom_n>
void tensor_core_mma_example()
{
    using namespace cute;
    using dtype = half_t;

    constexpr size_t m = atom_m * 16;
    constexpr size_t n = atom_n * 8;
    constexpr size_t k = 16;

    std::cout << "Tensor Core MMA Tiled MMA Example (M=" << m << ", N=" << n << ", K=" << k << ")\n";

    TiledMMA mma = make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{},
        Layout<Shape<Int<atom_m>, Int<atom_n>>>{}, // MMA Atoms layouts
        Tile<Int<m>, Int<n>, Int<k>>{} // Tile Layout
    );

    tiled_mma_example<m, n, k, dtype, decltype(mma)>();
}

int main()
{
    universal_fma_example<8, 6, 4>();
    tensor_core_mma_example<2, 2>();
}
