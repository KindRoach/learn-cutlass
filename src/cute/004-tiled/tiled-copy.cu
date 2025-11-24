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
    Tensor tile_S = local_tile(S, cta_tiler, cta_coord); // (BLK_M,BLK_N)
    Tensor tile_D = local_tile(D, cta_tiler, cta_coord); // (BLK_M,BLK_N)

    // Construct a Tensor corresponding to each thread's slice.
    ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    Tensor thr_tile_S = thr_copy.partition_S(tile_S); // (BLK_M/Thr_M * BLK_N/Thr_N)
    Tensor thr_tile_D = thr_copy.partition_D(tile_D); // (BLK_M/Thr_M * BLK_N/Thr_N)

    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_fragment because the first mode is the instruction-local mode
    Tensor fragment = make_fragment_like(thr_tile_D); // (BLK_M/Thr_M * BLK_N/Thr_N)

    // Copy: GMEM -> RMEM -> GMEM
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

        print_with_label("CTA Tile S: ", tile_S);
        print_with_label("CTA Tile D: ", tile_D);
        print_with_label("Thread Tile S: ", thr_tile_S);
        print_with_label("Thread Tile D: ", thr_tile_D);

        printf("=================== KERNEL END ==================\n\n");
    }
}

namespace
{
    using namespace cute;

    template <bool vectorized, bool coalesced, typename T>
    struct CopyConfig;

    template <typename T>
    struct CopyConfig<true, true, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<uint_byte_t<sizeof(T) * 4>>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{}));
        using ValLayout = decltype(make_layout(make_shape(Int<1>{}, Int<4>{})));
    };

    template <typename T>
    struct CopyConfig<false, true, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<T>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{}));
        using ValLayout = decltype(make_layout(make_shape(Int<1>{}, Int<4>{})));
    };

    template <typename T>
    struct CopyConfig<true, false, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<uint_byte_t<sizeof(T) * 4>>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<8>{}, Int<32>{})));
        using ValLayout = decltype(make_layout(make_shape(Int<1>{}, Int<4>{})));
    };

    template <typename T>
    struct CopyConfig<false, false, T>
    {
        using CopyAtom = Copy_Atom<UniversalCopy<T>, T>;
        using ThrLayout = decltype(make_layout(make_shape(Int<8>{}, Int<32>{})));
        using ValLayout = decltype(make_layout(make_shape(Int<1>{}, Int<4>{})));
    };
}

template <typename T, bool vectorized, bool coalesced>
void tiled_copy(
    thrust::device_vector<T>& src,
    thrust::device_vector<T>& dst,
    size_t m, size_t n
)
{
    using namespace cute;

    // warp global tensors
    Shape global_tensor_shape = make_shape(m, n);
    Layout layout_src = make_layout(global_tensor_shape, LayoutRight{}); // row-major
    Layout layout_dst = make_layout(global_tensor_shape, LayoutRight{}); // row-major
    Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(src.data())), layout_src);
    Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(dst.data())), layout_dst);

    // block tile shape / cta tiler
    auto block_shape = make_shape(Int<8>{}, Int<128>{});
    if (not evenly_divides(global_tensor_shape, block_shape))
    {
        std::cerr << "Expected the block_shape to evenly divide the global tensor shape." << std::endl;
        exit(1);
    }

    using copy_config = CopyConfig<vectorized, coalesced, T>;
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
        std::string latex_file = "tiled_copy_"
            + std::string(vectorized ? "vec" : "novec") + "_"
            + std::string(coalesced ? "coal" : "nocoal") + ".latex";
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
    std::vector<dtype> h_src(size);
    random_fill(h_src);

    d_vec d_src = h_src;
    d_vec d_dst(size);

    using func_t = std::function<void(d_vec&, d_vec&, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {"tiled_copy (vectorized, coalesced)", tiled_copy<dtype, true, true>},
        {"tiled_copy (non-vectorized, coalesced)", tiled_copy<dtype, false, true>},
        {"tiled_copy (vectorized, non-coalesced)", tiled_copy<dtype, true, false>},
        {"tiled_copy (non-vectorized, non-coalesced)", tiled_copy<dtype, false, false>},
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
        cuda_acc_check(h_src, d_dst);
    }
}
