#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cute/utils.cuh"


/// Vectorized copy kernel.
///
/// Uses `make_tiled_copy()` to perform a copy using vector instructions. This operation
/// has the precondition that pointers are aligned to the vector size.
///
template <class TensorS, class TensorD, class CtaTiler, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, CtaTiler cta_tiler, Tiled_Copy tiled_copy)
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

/// Main function
int main()
{
    using namespace cute;
    using Element = float;

    // Global data shape
    auto tensor_shape = make_shape(8, 16);
    thrust::host_vector<Element> h_S(size(tensor_shape));
    thrust::host_vector<Element> h_D(size(tensor_shape));

    // Init values
    for (size_t i = 0; i < h_S.size(); ++i)
    {
        h_S[i] = static_cast<Element>(i);
        h_D[i] = Element{};
    }

    thrust::device_vector<Element> d_S = h_S;
    thrust::device_vector<Element> d_D = h_D;

    // warp global tensors
    Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape));
    Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), make_layout(tensor_shape));
    print_tensor_with_label("tensor_S: ", make_tensor(h_S.data(), tensor_shape));

    // Define a static block shape (also called cta shape) [M, N].
    auto cta_tiler = make_shape(Int<4>{}, Int<8>{});
    print_with_label("cta_tiler: ", cta_tiler);

    // Check that the block shape evenly divides the tensor shape.
    if (not evenly_divides(tensor_shape, cta_tiler))
    {
        std::cerr << "Expected the block_shape to evenly divide the tensor shape." << std::endl;
        return -1;
    }

    // Construct a TiledCopy with a specific access pattern.
    //   This version uses a
    //   (1) Layout-of-Threads to describe the number and arrangement of threads (e.g. row-major, col-major, etc),
    //   (2) Layout-of-Values that each thread will access.

    // Thread arrangement
    Layout thr_layout = make_layout(make_shape(Int<1>{}, Int<8>{})); // (1,8) -> thr_idx
    Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{})); // (4,1) -> val_idx

    // Define `AccessType` which controls the size of the actual memory access instruction.
    // Copy multiple elements at a time for vectorized memory access.
    using CopyOp = UniversalCopy<uint_byte_t<sizeof(Element) * size(val_layout)>>;
    //using CopyOp = UniversalCopy<cutlass::AlignedArray<Element, size(val_layout)>>;  // A more generic type that supports many copy strategies
    //using CopyOp = AutoVectorizingCopy;                                              // An adaptable-width instruction that assumes maximal alignment of inputs

    // A Copy_Atom corresponds to one CopyOperation applied to Tensors of type Element.
    using Atom = Copy_Atom<CopyOp, Element>;

    // Construct tiled copy, a tiling of copy atoms.
    //
    // Note, this assumes the vector and thread layouts are aligned with contiguous data
    // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
    // reads. Alternative value layouts are also possible, though incompatible layouts
    // will result in compile time errors.
    TiledCopy tiled_copy = make_tiled_copy(
        Atom{}, // Access strategy
        thr_layout, // thread layout
        val_layout); // value layout
    print_with_label("tiled_copy: ", tiled_copy);

    // cal gird and block dimensions
    dim3 gridDim(
        size<0>(ceil_div(tensor_shape, cta_tiler)),
        size<1>(ceil_div(tensor_shape, cta_tiler))
    );
    dim3 blockDim(size(thr_layout));
    printf("Grid: (%d, %d), Block: %d\n", gridDim.x, gridDim.y, blockDim.x);

    // Launch the kernel
    copy_kernel_vectorized<<< gridDim, blockDim >>>(
        tensor_S,
        tensor_D,
        cta_tiler,
        tiled_copy);

    cudaDeviceSynchronize();
    h_D = d_D;
    print_tensor_with_label("tensor_D: ", make_tensor(h_D.data(), tensor_shape));
}
