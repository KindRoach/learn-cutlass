#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void print_layout_with_label(const char* label, const auto& layout)
{
    using namespace cute;
    print(label);
    print_layout(layout);
    print("\n");
}

void print_copy_with_label(const char* label, const auto& copy)
{
    using namespace cute;
    print(label);
    print(copy);
    print("\n");
}

void print_mma_with_label(const char* label, const auto& mma)
{
    using namespace cute;
    print(label);
    print(mma);
    print("\n");
}

__host__ __device__
void print_tensor_with_label(const char* label, const auto& tensor)
{
    using namespace cute;
    print(label);
    print_tensor(tensor);
    print("\n");
}

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
    if (thread0())
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
        __syncthreads();
    }

    // Epilogue
    axpby(alpha, tCrC, beta, tCgC);
}

template <
    class T,
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

    // all in row-major
    // c [m,n] = a [m,k] x b [k,n]
    auto prob_shape = make_shape(m, n, k);

    // device memory layouts (dynamic)
    auto mA_layout = make_layout(make_shape(m, k), LayoutRight{});
    auto mB_layout = make_layout(make_shape(n, k), LayoutLeft{});
    auto mC_layout = make_layout(make_shape(m, n), LayoutRight{});

    // cta tile (static)
    auto cta_tiler = Shape<Int<bM>, Int<bN>, Int<bK>>{};
    std::cout << "CTA Tiler: " << cta_tiler << "\n\n";

    // Define the smem layouts (static)
    auto sA_layout = Layout<Shape<Int<bM>, Int<bK>>>{};
    auto sB_layout = Layout<Shape<Int<bN>, Int<bK>>>{};

    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        Layout<Shape<_32, _8>>{}, // Thr layout 32x8 m-major
        Layout<Shape<_1, _2>>{} // Val layout  4x1 m-major
    );

    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<UniversalCopy<T>, T>{},
        Layout<Shape<_32, _8>>{}, // Thr layout 32x8 n-major
        Layout<Shape<_1, _2>>{} // Val layout  4x1 n-major
    );

    TiledMMA mmaC = make_tiled_mma(
        UniversalFMA<T>{},
        Layout<Shape<_16, _16, _1>>{} // 16x16x1 UniversalFMA
    );

    // Debug info
    print_layout_with_label("Layout mA: ", mA_layout);
    print_layout_with_label("Layout mB: ", mB_layout);
    print_layout_with_label("Layout mC: ", mC_layout);

    print_layout_with_label("Layout sA: ", sA_layout);
    print_layout_with_label("Layout sB: ", sB_layout);

    print_copy_with_label("copyA: ", copyA);
    print_copy_with_label("copyB: ", copyB);
    print_mma_with_label("mmaC: ", mmaC);

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
    cudaDeviceSynchronize();

    // copy back to host and print result
    thrust::host_vector<T> host_c = c;
    auto mC = make_tensor(host_c.data(), mC_layout);
    print_tensor_with_label("Result mC: ", mC);
}

int main()
{
    using dtype = float;
    using d_vec = thrust::device_vector<dtype>;

    size_t m = 64, n = 96, k = 128;
    constexpr size_t bM = 32, bN = 32, bK = 16;
    constexpr size_t tM = 32, tN = 32, tK = 8;
    constexpr size_t tCM = 16, tCN = 16;

    std::vector<dtype> a(m * k), b(k * n);

    for (int i = 0; i < a.size(); ++i)
        a[i] = static_cast<dtype>(i % 255);

    for (int i = 0; i < b.size(); ++i)
        b[i] = static_cast<dtype>(i % 127);

    d_vec d_a = a;
    d_vec d_b = b;
    d_vec d_c(m * n);

    cute_gemm<
        dtype,
        bM, bN, bK,
        tM, tN, tK,
        tCM, tCN
    >(d_a, d_b, d_c, m, n, k);
}
