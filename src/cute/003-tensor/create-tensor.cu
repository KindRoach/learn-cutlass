#include <cute/tensor.hpp>

template <typename T>
__global__ void shared_mem_nonowning_tensor_example()
{
    using namespace cute;

    // Shared memory (static or dynamic layouts)
    Layout smem_layout = make_layout(make_shape(Int<4>{}, Int<8>{}));
    __shared__ T smem[decltype(cosize(smem_layout))::value]; // (static-only allocation)
    Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
    Tensor smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout), LayoutRight{});

    print("smem_4x8_col = ");
    print(smem_4x8_col);
    print("\n");

    print("smem_4x8_row = ");
    print(smem_4x8_row);
    print("\n");
}

template <typename T>
void nonowning_tensor_example()
{
    using namespace cute;

    T* A;

    cudaMalloc(&A, 1024 * 1024); // 1MB

    // Untagged pointers (static or dynamic layouts)
    Tensor tensor_8 = make_tensor(A, make_layout(Int<8>{}));
    Tensor tensor_8s = make_tensor(A, Int<8>{});
    Tensor tensor_8d2 = make_tensor(A, 8, 2);

    print("tensor_8 = ");
    print(tensor_8);
    print("\n");

    print("tensor_8s = ");
    print(tensor_8s);
    print("\n");

    print("tensor_8d2 = ");
    print(tensor_8d2);
    print("\n");

    // Global memory (static or dynamic layouts)
    Tensor gmem_8s = make_tensor(make_gmem_ptr(A), Int<8>{});
    Tensor gmem_8d = make_tensor(make_gmem_ptr(A), 8);
    Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{}, 16));
    Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape(8, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));

    print("gmem_8s = ");
    print(gmem_8s);
    print("\n");

    print("gmem_8d = ");
    print(gmem_8d);
    print("\n");

    print("gmem_8sx16d = ");
    print(gmem_8sx16d);
    print("\n");

    print("gmem_8dx16s = ");
    print(gmem_8dx16s);
    print("\n");

    cudaFree(A);

    shared_mem_nonowning_tensor_example<T><<<1,1>>>();
    cudaDeviceSynchronize();
}

template <typename T>
void owning_tensor_example()
{
    using namespace cute;

    // Register memory (static layouts only)
    Tensor rmem_4x8_col = make_tensor<float>(Shape<_4, _8>{});
    Tensor rmem_4x8_row = make_tensor<float>(Shape<_4, _8>{}, LayoutRight{});
    Tensor rmem_4x8_pad = make_tensor<float>(Shape<_4, _8>{}, Stride<_32, _2>{});
    Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);

    print("rmem_4x8_col = ");
    print(rmem_4x8_col);
    print("\n");

    print("rmem_4x8_row = ");
    print(rmem_4x8_row);
    print("\n");

    print("rmem_4x8_pad = ");
    print(rmem_4x8_pad);
    print("\n");

    print("rmem_4x8_like = ");
    print(rmem_4x8_like);
    print("\n");
}

int main()
{
    using namespace cute;
    nonowning_tensor_example<float>();
    owning_tensor_example<float>();
}
