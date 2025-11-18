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
    class AGlobalLayout, class ASmemLayout, class AThreadLayout,
    class BGlobalLayout, class BSmemLayout, class BThreadLayout,
    class CGlobalLayout, class CThreadLayout
>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void cute_gemm_kernel(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    T const* A, AGlobalLayout mA_layout, ASmemLayout sA_layout, AThreadLayout tA,
    T const* B, BGlobalLayout mB_layout, BSmemLayout sB_layout, BThreadLayout tB,
    T* C, CGlobalLayout mC_layout, CThreadLayout tC,
    T alpha, T beta)
{
    using namespace cute;

    // Represent the full tensors on global memory
    Tensor mA = make_tensor(make_gmem_ptr(A), mA_layout); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), mB_layout); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), mC_layout); // (M,N)

    // Get the tile for this thread block
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
    Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M,THR_K,k)
    Tensor tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N,THR_K,k)
    Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M,THR_K)
    Tensor tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N,THR_K)

    // For gemm
    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); // (THR_M,BLK_K)
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); // (THR_N,BLK_K)
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)
    Tensor tCrC = make_tensor_like(tCgC); // (THR_M,THR_N)

    if (cbu::is_debug and thread(1))
    {
        printf("=================== CUTE GEMM KERNEL START ==================\n\n");
        printf("Grid: (%d,%d), Block: (%d), Thread: (%d)\n\n",
               gridDim.x, gridDim.y,
               blockDim.x,
               threadIdx.x
        );

        print_tensor_with_label("mA : ", mA);
        print_tensor_with_label("gA : ", gA);
        print_tensor_with_label("sA : ", sA);
        print_tensor_with_label("tAgA : ", tAgA);
        print_tensor_with_label("tAsA : ", tAsA);

        print_tensor_with_label("mB : ", mB);
        print_tensor_with_label("gB : ", gB);
        print_tensor_with_label("sB : ", sB);
        print_tensor_with_label("tBgB : ", tBgB);
        print_tensor_with_label("tBsB : ", tBsB);

        print_tensor_with_label("mC : ", mC);
        print_tensor_with_label("gC : ", gC);
        print_tensor_with_label("tCsA : ", tCsA);
        print_tensor_with_label("tCsB : ", tCsB);
        print_tensor_with_label("tCgC : ", tCgC);
        print_tensor_with_label("tCrC : ", tCrC);
        printf("=================== CUTE GEMM KERNEL END ====================\n\n");
    }

    // Mainloop
    auto K_TILE_MAX = size<2>(tAgA);
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        copy(tAgA(_, _, k_tile), tAsA); // A   (THR_M,THR_K) -> (THR_M,THR_K)
        copy(tBgB(_, _, k_tile), tBsB); // B   (THR_N,THR_K) -> (THR_N,THR_K)

        cp_async_fence(); // Label the end of (potential) cp.async instructions
        cp_async_wait<0>(); // Sync on all (potential) cp.async instructions
        __syncthreads(); // Wait for all threads to write to smem

        gemm(tCsA, tCsB, tCrC); // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)
        __syncthreads(); // Wait for all threads to read from smem
    }

    // Epilogue
    axpby(alpha, tCrC, beta, tCgC);
}

template <
    class T, cbu::matrix_layout b_layout,
    size_t bM, size_t bN, size_t bK,
    size_t tM, size_t tN, size_t tK,
    size_t tCM, size_t tCN
>
void cute_gemm(
    thrust::device_vector<T>& a,
    thrust::device_vector<T>& b,
    thrust::device_vector<T>& c,
    size_t m, size_t n, size_t k
)
{
    using namespace cute;

    cbu::check_divisible(m, bM, "M must be divisible by bM");
    cbu::check_divisible(n, bN, "N must be divisible by bN");
    cbu::check_divisible(k, bK, "K must be divisible by bK");

    auto prob_shape = make_shape(m, n, k);

    // device memory layouts (dynamic)
    using BLayout = std::conditional_t<
        b_layout == cbu::matrix_layout::row_major,
        LayoutLeft,
        LayoutRight
    >;
    auto mA_layout = make_layout(make_shape(m, k), LayoutRight{});
    auto mB_layout = make_layout(make_shape(n, k), BLayout{});
    auto mC_layout = make_layout(make_shape(m, n), LayoutRight{});

    // cta tile (static)
    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});

    // Define the smem layouts (static)
    auto sA_layout = make_layout(make_shape(Int<bM>{}, Int<bK>{}));
    // use BLayout to avoid bank conflict
    auto sB_layout = make_layout(make_shape(Int<bN>{}, Int<bK>{}), BLayout{});

    // Define the thread layouts (static)
    auto tA_layout = make_layout(make_shape(Int<tM>{}, Int<tK>{}));
    // use BLayout to avoid un-coalesced access
    auto tB_layout = make_layout(make_shape(Int<tN>{}, Int<tK>{}), BLayout{});
    auto tC_layout = make_layout(make_shape(Int<tCM>{}, Int<tCN>{}));

    // Validate that the thread tile sizes are the same
    static_assert(size(tA_layout) == size(tB_layout));
    static_assert(size(tA_layout) == size(tC_layout));

    // Validate that the tiler sizes are multiples of the thread tile sizes
    static_assert(size<0>(cta_tiler) % size<0>(tA_layout) == _0{});
    static_assert(size<2>(cta_tiler) % size<1>(tA_layout) == _0{});
    static_assert(size<1>(cta_tiler) % size<0>(tB_layout) == _0{});
    static_assert(size<2>(cta_tiler) % size<1>(tB_layout) == _0{});
    static_assert(size<0>(cta_tiler) % size<0>(tC_layout) == _0{});
    static_assert(size<1>(cta_tiler) % size<1>(tC_layout) == _0{});

    // launch kernel
    dim3 dimBlock(size(tA_layout));
    dim3 dimGrid(size(ceil_div(m, Int<bM>{})), size(ceil_div(n, Int<bN>{})));
    cute_gemm_kernel<<<dimGrid, dimBlock>>>(
        prob_shape, cta_tiler,
        a.data().get(), mA_layout, sA_layout, tA_layout,
        b.data().get(), mB_layout, sB_layout, tB_layout,
        c.data().get(), mC_layout, tC_layout,
        static_cast<T>(1.0), static_cast<T>(0.0)
    );
    cudaDeviceSynchronize();

    if (cbu::is_debug)
    {
        std::cout << "CTA Tiler: " << cta_tiler << "\n\n";

        print_layout_with_label("Layout mA: ", mA_layout);
        print_layout_with_label("Layout mB: ", mB_layout);
        print_layout_with_label("Layout mC: ", mC_layout);

        print_layout_with_label("Layout sA: ", sA_layout);
        print_layout_with_label("Layout sB: ", sB_layout);

        print_layout_with_label("Layout tA: ", tA_layout);
        print_layout_with_label("Layout tB: ", tB_layout);
        print_layout_with_label("Layout tC: ", tC_layout);

        thrust::host_vector<T> host_c = c;
        auto mC = make_tensor(host_c.data(), mC_layout);
        print_tensor_with_label("Result mC: ", mC);
    }
}

template <cbu::matrix_layout b_layout>
void test_matrix_multiply()
{
    using namespace cbu;

    std::string b_major = b_layout == matrix_layout::row_major ? "row major" : "col major";
    std::cout << "-------------- matrix b in " << b_major << " --------------\n";

    using dtype = float;
    using d_vec = thrust::device_vector<dtype>;
    size_t secs = is_debug ? 0 : 10;

    // // small case
    // size_t m = 8, n = 12, k = 16;
    // constexpr size_t bM = 4, bN = 4, bK = 8;
    // constexpr size_t tM = 2, tN = 2, tK = 8;
    // constexpr size_t tCM = 4, tCN = 4;

    // // large case
    // size_t m = 64, n = 96, k = 128;
    // constexpr size_t bM = 32, bN = 32, bK = 16;
    // constexpr size_t tM = 32, tN = 32, tK = 8;
    // constexpr size_t tCM = 16, tCN = 16;

    // huge case
    size_t m = 2 * 1024, n = 512, k = 1024;
    constexpr size_t bM = 32, bN = 32, bK = 16;
    constexpr size_t tM = 32, tN = 32, tK = 8;
    constexpr size_t tCM = 16, tCN = 16;

    std::vector<dtype> a(m * k), b(k * n), c(m * n);

    for (int i = 0; i < a.size(); ++i)
        a[i] = static_cast<dtype>(i % 255);

    for (int i = 0; i < b.size(); ++i)
        b[i] = static_cast<dtype>(i % 127);

    d_vec d_a = a;
    d_vec d_b = b;
    d_vec d_c(m * n);

    std::cout << "matrix_transpose_ref:\n";
    benchmark_func_by_time(secs, [&]
    {
        matrix_multiply_ref<dtype, b_layout>(a, b, c, m, n, k);
    });

    using func_t = std::function<void(d_vec&, d_vec&, d_vec&, size_t, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {
            "cute_gemm", cute_gemm<
                dtype, b_layout,
                bM, bN, bK,
                tM, tN, tK,
                tCM, tCN
            >
        },
    };

    for (const auto& [func_name,func] : funcs)
    {
        std::cout << "\n" << func_name << ":\n";
        fill(d_c.begin(), d_c.end(), 0);
        benchmark_func_by_time(
            secs,
            [&]()
            {
                func(d_a, d_b, d_c, m, n, k);
                cuda_check(cudaDeviceSynchronize());
            },
            {
                .total_mem_bytes = sizeof(dtype) * (m * k + k * n + m * n),
                .total_flop = 2 * m * n * k,
            }
        );
        cuda_acc_check(c, d_c);
    }
}

int main()
{
    test_matrix_multiply<cbu::matrix_layout::row_major>();
    test_matrix_multiply<cbu::matrix_layout::col_major>();
}
