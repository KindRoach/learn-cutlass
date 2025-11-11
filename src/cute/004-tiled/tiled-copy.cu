#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cpp-bench-utils/utils.hpp>
#include <cute/tensor.hpp>

#include "cute/utils.cuh"


template <class TensorS, class TensorD, class CtaTiler, class Tiled_Copy>
__global__ void tiled_copy_kernel(TensorS S, TensorD D, CtaTiler cta_tiler, Tiled_Copy tiled_copy)
{
    using namespace cute;

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y);
    Tensor tile_S = local_tile(S, cta_tiler, cta_coord); // (BLK_M,BLK_K,k)
    Tensor tile_D = local_tile(D, cta_tiler, cta_coord); // (BLK_N,BLK_K,k)

    // Construct a Tensor corresponding to each thread's slice.
    ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    Tensor thr_tile_S = thr_copy.partition_S(tile_S); // (CopyOp, CopyM, CopyN)
    Tensor thr_tile_D = thr_copy.partition_D(tile_D); // (CopyOp, CopyM, CopyN)

    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_fragment because the first mode is the instruction-local mode
    Tensor fragment = make_fragment_like(thr_tile_D); // (CopyOp, CopyM, CopyN)

    // Copy from GMEM to RMEM and from RMEM to GMEM
    copy(tiled_copy, thr_tile_S, fragment);
    copy(tiled_copy, fragment, thr_tile_D);
}

template <typename T, cbu::matrix_layout dst_layout>
void tiled_copy(
    thrust::device_vector<T>& src,
    thrust::device_vector<T>& dst,
    size_t m, size_t n
)
{
    using namespace cute;

    using DstLayout = std::conditional_t<
        dst_layout == cbu::matrix_layout::row_major,
        LayoutRight,
        LayoutLeft
    >;

    // warp global tensors
    Shape global_tensor_shape = make_shape(m, n);
    Layout layout_src = make_layout(global_tensor_shape, LayoutRight{}); // row-major
    Layout layout_dst = make_layout(global_tensor_shape, DstLayout{});
    Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(src.data())), layout_src);
    Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(dst.data())), layout_dst);

    // block tile shape / cta tiler
    auto block_shape = make_shape(Int<128>{}, Int<64>{});
    if (not evenly_divides(global_tensor_shape, block_shape))
    {
        std::cerr << "Expected the block_shape to evenly divide the global tensor shape." << std::endl;
        return;
    }

    auto thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}); // 256 threads
    auto val_layout = make_layout(make_shape(Int<1>{}, Int<4>{})); // each thread copy 4 elements at a time
    TiledCopy tiled_copy = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        thr_layout,
        val_layout
    );

    // cal gird and block dimensions
    dim3 gridDim(
        size<0>(ceil_div(global_tensor_shape, block_shape)),
        size<1>(ceil_div(global_tensor_shape, block_shape))
    );
    dim3 blockDim(size(tiled_copy));

    // Launch the kernel
    tiled_copy_kernel<<< gridDim, blockDim >>>(
        tensor_S,
        tensor_D,
        block_shape,
        tiled_copy);
}

template <cbu::matrix_layout dst_layout>
void test_matrix_copy()
{
    using namespace cbu;

    std::string dst_major = dst_layout == matrix_layout::row_major ? "row major" : "col major";
    std::cout << "-------------- matrix dst in " << dst_major << " --------------\n";

    using dtype = float;
    using d_vec = thrust::device_vector<dtype>;

    size_t secs = 10;
    size_t m = 20 * 1024, n = 5 * 1024; // 100M elements

    size_t size = m * n;
    std::vector<dtype> h_src(size), h_dst(size);
    random_fill(h_src);

    d_vec d_src = h_src;
    d_vec d_dst(size);

    std::cout << "copy_ref:\n";
    if (dst_layout == matrix_layout::col_major)
    {
        benchmark_func_by_time(secs, [&] { matrix_transpose_ref<dtype>(h_src, h_dst, m, n); });
    }
    else
    {
        benchmark_func_by_time(secs, [&] { std::copy(h_src.begin(), h_src.end(), h_dst.begin()); });
    }

    using func_t = std::function<void(d_vec&, d_vec&, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {"tiled_copy", tiled_copy<dtype, dst_layout>},
    };

    for (const auto& [func_name,func] : funcs)
    {
        std::cout << "\n" << func_name << ":\n";
        fill(d_dst.begin(), d_dst.end(), 0);
        benchmark_func_by_time(secs, [&]()
        {
            func(d_src, d_dst, m, n);
            cuda_check(cudaDeviceSynchronize());
        });
        cuda_acc_check(h_dst, d_dst);
    }
}

/// Main function
int main()
{
    test_matrix_copy<cbu::matrix_layout::row_major>();
    // test_matrix_copy<cbu::matrix_layout::col_major>();
}
