#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cpp-bench-utils/utils.hpp>
#include <cute/tensor.hpp>

#include "cute/utils.cuh"

// In  : [m,n] in row-major
// Out : [n,m] in row-major

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

    if (cbu::is_debug and thread(1))
    {
        printf("=================== KERNEL START ==================\n\n");
        printf("Grid: (%d,%d), Block: (%d), Thread: (%d)\n\n",
               gridDim.x, gridDim.y,
               blockDim.x,
               threadIdx.x
        );

        print_tensor_with_label("CTA Tile S: ", tile_S);
        print_tensor_with_label("CTA Tile D: ", tile_D);
        print_tensor_with_label("Thread Tile S: ", thr_tile_S);
        print_tensor_with_label("Thread Tile D: ", thr_tile_D);

        printf("=================== KERNEL END ==================\n\n");
    }
}


namespace
{
    using namespace cute;

    template <bool row_major, bool t8x32, bool v1x4, typename T>
    struct CopyConfig;

    template <typename T>
    struct CopyConfig<true, true, true, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<T>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{}));
        using ValLayout = decltype(make_layout(make_shape(Int<1>{}, Int<4>{})));
    };

    template <typename T>
    struct CopyConfig<true, true, false, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<T>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{}));
        using ValLayout = decltype(make_layout(make_shape(Int<4>{}, Int<1>{})));
    };

    template <typename T>
    struct CopyConfig<true, false, true, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<T>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}));
        using ValLayout = decltype(make_layout(make_shape(Int<1>{}, Int<4>{})));
    };

    template <typename T>
    struct CopyConfig<true, false, false, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<T>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{}));
        using ValLayout = decltype(make_layout(make_shape(Int<4>{}, Int<1>{})));
    };

    template <typename T>
    struct CopyConfig<false, true, true, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<T>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<8>{}, Int<32>{})));
        using ValLayout = decltype(make_layout(make_shape(Int<1>{}, Int<4>{})));
    };

    template <typename T>
    struct CopyConfig<false, true, false, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<T>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<8>{}, Int<32>{})));
        using ValLayout = decltype(make_layout(make_shape(Int<4>{}, Int<1>{})));
    };

    template <typename T>
    struct CopyConfig<false, false, true, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<T>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<32>{}, Int<8>{})));
        using ValLayout = decltype(make_layout(make_shape(Int<1>{}, Int<4>{})));
    };

    template <typename T>
    struct CopyConfig<false, false, false, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<T>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<32>{}, Int<8>{})));
        using ValLayout = decltype(make_layout(make_shape(Int<4>{}, Int<1>{})));
    };
}

template <typename T, bool row_major, bool t8x32, bool v1x4>
void transpose_tiled_copy(
    thrust::device_vector<T>& src,
    thrust::device_vector<T>& dst,
    size_t m, size_t n
)
{
    using namespace cute;

    // warp global tensors
    Shape global_tensor_shape = make_shape(m, n);
    Layout layout_src = make_layout(global_tensor_shape, LayoutRight{}); // row-major
    Layout layout_dst = make_layout(global_tensor_shape, LayoutLeft{}); // col-major as transpose output
    Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(src.data())), layout_src);
    Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(dst.data())), layout_dst);

    // block tile shape / cta tiler
    auto block_shape = make_shape(Int<128>{}, Int<128>{});
    if (not evenly_divides(global_tensor_shape, block_shape))
    {
        std::cerr << "Expected the block_shape to evenly divide the global tensor shape." << std::endl;
        exit(1);
    }

    using copy_config = CopyConfig<row_major, t8x32, v1x4, T>;
    TiledCopy tiled_copy = make_tiled_copy(
        typename copy_config::CopyAtom{},
        typename copy_config::ThrLayout{},
        typename copy_config::ValLayout{}
    );
    if (not evenly_divides(block_shape, typename decltype(tiled_copy)::Tiler_MN{}))
    {
        std::cerr << "Expected the tiled copy tiler to evenly divide the block_shape." << std::endl;
        exit(1);
    }

    if (cbu::is_debug)
    {
        std::string latex_file = std::string("tiled_copy_")
            + (row_major ? "row_major_" : "col_major_")
            + (t8x32 ? "t8x32_" : "t32x8_")
            + (v1x4 ? "v1x4" : "v4x1")
            + ".latex";
        print_latex_to_file(latex_file, tiled_copy);
        print_with_label("Tiled Copy: ", tiled_copy);
    }

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

int main()
{
    using namespace cbu;

    using dtype = float;
    using d_vec = thrust::device_vector<dtype>;

    size_t secs = is_debug ? 0 : 10;
    size_t m = 20 * 1024, n = 5 * 1024; // 100M elements

    size_t size = m * n;
    std::vector<dtype> h_src(size), h_dst(size);
    random_fill(h_src);

    d_vec d_src = h_src;
    d_vec d_dst(size);

    std::cout << "matrix_transpose_ref:\n";
    benchmark_func_by_time(secs, [&] { matrix_transpose_ref(h_src, h_dst, m, n); });

    using func_t = std::function<void(d_vec&, d_vec&, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {
            "transpose_single_tiled_copy_row_major_t8x32_v1x4",
            transpose_tiled_copy<dtype, true, true, true>
        },
        {
            "transpose_single_tiled_copy_row_major_t8x32_v4x1",
            transpose_tiled_copy<dtype, true, true, false>
        },
        {
            "transpose_single_tiled_copy_row_major_t32x8_v1x4",
            transpose_tiled_copy<dtype, true, false, true>
        },
        {
            "transpose_single_tiled_copy_row_major_t32x8_v4x1",
            transpose_tiled_copy<dtype, true, false, false>
        },
        {
            "transpose_single_tiled_copy_col_major_t8x32_v1x4",
            transpose_tiled_copy<dtype, false, true, true>
        },
        {
            "transpose_single_tiled_copy_col_major_t8x32_v4x1",
            transpose_tiled_copy<dtype, false, true, false>
        },
        {
            "transpose_single_tiled_copy_col_major_t32x8_v1x4",
            transpose_tiled_copy<dtype, false, false, true>
        },
        {
            "transpose_single_tiled_copy_col_major_t32x8_v4x1",
            transpose_tiled_copy<dtype, false, false, false>
        },
    };

    for (const auto& [func_name,func] : funcs)
    {
        std::cout << "\n" << func_name << ":\n";
        fill(d_dst.begin(), d_dst.end(), 0);
        benchmark_func_by_time(
            secs,
            [&]
            {
                func(d_src, d_dst, m, n);
                cuda_check(cudaDeviceSynchronize());
            },
            {
                .total_mem_bytes = sizeof(dtype) * size * 2
            }
        );
        cuda_acc_check(h_dst, d_dst);
    }
}
