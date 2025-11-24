#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cpp-bench-utils/utils.hpp"
#include "cute/utils.cuh"

// A : [m,k] in row-major
// B : [k,n] in row-major or col-major
// C = A x B : [m,n] in row-major

template <
    class T, class ProblemShape, class CtaTiler,
    class AGlobalLayout, class ASmemLayout, class TiledCopyA,
    class BGlobalLayout, class BSmemLayout, class TiledCopyB,
    class CGlobalLayout, class TiledMmaC
>
__global__ static
__launch_bounds__(decltype(size(TiledMmaC{}))::value)
void cute_gemm_kernel(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    T const* A, AGlobalLayout mA_layout, ASmemLayout sA_layout, TiledCopyA copy_a,
    T const* B, BGlobalLayout mB_layout, BSmemLayout sB_layout, TiledCopyB copy_b,
    T* C, CGlobalLayout mC_layout, TiledMmaC mma_c,
    T alpha, T beta)
{
    using namespace cute;

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), mA_layout); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), mB_layout); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), mC_layout); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ T smemA[cosize_v<ASmemLayout>];
    __shared__ T smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N,BLK_K)

    // For copy
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K)
    Tensor tArA = make_fragment_like(tAsA); // (CPY,CPY_M,CPY_K)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K)
    Tensor tBrB = make_fragment_like(tBsB); // (CPY,CPY_N,CPY_K)

    // For gemm
    ThrMMA thr_mma = mma_c.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)

    // Clear the accumulators
    clear(tCrC);

    // Debug info
    if (cbu::is_debug and thread(1))
    {
        printf("=================== CUTE GEMM KERNEL START ==================\n\n");
        printf("Grid: (%d,%d), Block: (%d), Thread: (%d)\n\n",
               gridDim.x, gridDim.y,
               blockDim.x,
               threadIdx.x
        );

        print_with_label("mA : ", mA);
        print_with_label("gA : ", gA);
        print_with_label("sA : ", sA);
        print_with_label("tAgA : ", tAgA);
        print_with_label("tAsA : ", tAsA);

        print_with_label("mB : ", mB);
        print_with_label("gB : ", gB);
        print_with_label("sB : ", sB);
        print_with_label("tBgB : ", tBgB);
        print_with_label("tBsB : ", tBsB);

        print_with_label("mC : ", mC);
        print_with_label("gC : ", gC);
        print_with_label("tCsA : ", tCsA);
        print_with_label("tCsB : ", tCsB);
        print_with_label("tCgC : ", tCgC);
        print_with_label("tCrC : ", tCrC);
        printf("=================== CUTE GEMM KERNEL END ====================\n\n");
    }

    // Mainloop
    auto K_TILE_MAX = size<3>(tAgA);
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        // copy gmem to rmem
        copy(copy_a, tAgA(_, _, _, k_tile), tArA);
        copy(copy_b, tBgB(_, _, _, k_tile), tBrB);
        __syncthreads();

        // copy rmem to smem
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();

        // Compute gemm on mma-partitioned smem
        gemm(mma_c, tCsA, tCsB, tCrC);

        // No __syncthreads() need here until next copy to smem
    }

    // Epilogue
    axpby(alpha, tCrC, beta, tCgC);
}

template <class T, cbu::matrix_layout b_layout>
void matrix_multiply_tiled_mma(
    thrust::device_vector<T>& a,
    thrust::device_vector<T>& b,
    thrust::device_vector<T>& c,
    size_t m, size_t n, size_t k
)
{
    using namespace cute;
    using namespace cbu;

    // all in row-major
    // c [m,n] = a [m,k] x b [k,n]
    auto prob_shape = make_shape(m, n, k);

    // device memory layouts (dynamic)
    using BStride = std::conditional_t<
        b_layout == matrix_layout::row_major,
        LayoutLeft,
        LayoutRight
    >;
    auto mA_layout = make_layout(make_shape(m, k), LayoutRight{}); // row-major
    auto mB_layout = make_layout(make_shape(n, k), BStride{}); // row or col-major
    auto mC_layout = make_layout(make_shape(m, n), LayoutRight{}); // row-major

    constexpr size_t bM = 128, bN = 128, bK = 8;

    // cta tile (static)
    auto cta_tiler = Shape<Int<bM>, Int<bN>, Int<bK>>{};

    // Define the smem layouts (static)
    auto sA_layout = make_layout(make_shape(Int<bM>{}, Int<bK>{}), LayoutRight{});
    auto sB_layout = make_layout(make_shape(Int<bN>{}, Int<bK>{}), BStride{});

    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        make_layout(make_shape(_32{}, _8{}), LayoutRight{}),
        make_layout(make_shape(_1{}, _1{}))
    );

    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        make_layout(make_shape(_32{}, _8{}), BStride{}),
        make_layout(make_shape(_1{}, _1{}))
    );

    if (not evenly_divides(shape(sA_layout), typename decltype(copyA)::Tiler_MN{})
        or not evenly_divides(shape(sB_layout), typename decltype(copyB)::Tiler_MN{}))
    {
        std::cerr << "Expected the tiled copy tiler to evenly divide the smem_shape." << std::endl;
        exit(1);
    }

    TiledMMA mmaC = make_tiled_mma(
        UniversalFMA<T>{},
        Layout<Shape<_16, _16, _1>>{} // 16x16x1 UniversalFMA
    );

    if (size(mmaC) != size(typename decltype(copyA)::TiledNumThr{})
        or size(mmaC) != size(typename decltype(copyB)::TiledNumThr{}))
    {
        std::cerr << "Expected the tiled_copy thread num equals tiled_mma's" << std::endl;
        exit(1);
    }

    if (is_debug)
    {
        std::cout << "CTA Tiler: " << cta_tiler << "\n\n";

        print_with_label("Layout mA: ", mA_layout);
        print_with_label("Layout mB: ", mB_layout);
        print_with_label("Layout mC: ", mC_layout);

        print_with_label("Layout sA: ", sA_layout);
        print_with_label("Layout sB: ", sB_layout);

        print_with_label("copyA: ", copyA);
        print_with_label("copyB: ", copyB);
        print_with_label("mmaC: ", mmaC);
    }

    // launch kernel
    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(m, Int<bM>{})), size(ceil_div(n, Int<bN>{})));
    cute_gemm_kernel<<<dimGrid, dimBlock>>>(
        prob_shape, cta_tiler,
        a.data().get(), mA_layout, sA_layout, copyA,
        b.data().get(), mB_layout, sB_layout, copyB,
        c.data().get(), mC_layout, mmaC,
        static_cast<T>(1.0), static_cast<T>(0.0)
    );
}

template <cbu::matrix_layout b_layout>
void test_matrix_multiply()
{
    using namespace cbu;

    std::string b_major = b_layout == matrix_layout::row_major ? "row major" : "col major";
    std::cout << "-------------- matrix b in " << b_major << " --------------\n";

    using dtype = cute::half_t;
    using d_vec = thrust::device_vector<dtype>;
    size_t secs = is_debug ? 0 : 10;

    // huge case
    size_t m = 2 * 1024, n = 512, k = 1024;

    std::vector<dtype> a(m * k), b(k * n), c(m * n);
    random_fill(a);
    random_fill(b);

    std::cout << "matrix_multiply_ref:\n";
    BenchmarkOptions opt{
        .total_mem_bytes = (m * k + k * n + m * n) * sizeof(dtype),
        .total_flop = 2 * m * n * k,
    };
    benchmark_func_by_time(secs, [&]
    {
        matrix_multiply_ref<dtype, b_layout>(a, b, c, m, n, k);
    }, opt);

    d_vec d_a = a;
    d_vec d_b = b;
    d_vec d_c(c.size());

    using func_t = std::function<void(d_vec&, d_vec&, d_vec&, size_t, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {"matrix_multiply_tiled_mma", matrix_multiply_tiled_mma<dtype, b_layout>},
    };

    for (const auto& [func_name,func] : funcs)
    {
        std::cout << "\n" << func_name << ":\n";
        fill(d_c.begin(), d_c.end(), static_cast<dtype>(0));
        benchmark_func_by_time(secs, [&]()
        {
            func(d_a, d_b, d_c, m, n, k);
            cuda_check(cudaDeviceSynchronize());
        }, opt);
        cuda_acc_check(c, d_c);
    }
}

int main()
{
    test_matrix_multiply<cbu::matrix_layout::row_major>();
    test_matrix_multiply<cbu::matrix_layout::col_major>();
}
