#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "util.cuh"

int main()
{
    using dtype = cutlass::half_t;
    int M = 128, N = 128, K = 128;
    dtype alpha{1.0f}, beta{0.0f};
    auto [A, B, C, C_ref] =
        learn_util::create_random_tensors<dtype, cutlass::layout::RowMajor>(M, N, K, 42);

    using Gemm = cutlass::gemm::device::Gemm<
        dtype, cutlass::layout::RowMajor,
        dtype, cutlass::layout::RowMajor,
        dtype, cutlass::layout::RowMajor>;

    Gemm gemm_op;
    Gemm::Arguments args(
        {M, N, K},
        {A.device_data(), A.stride(0)},
        {B.device_data(), B.stride(0)},
        {C.device_data(), C.stride(0)},
        {C.device_data(), C.stride(0)},
        {alpha, beta});
    cutlass::Status status = gemm_op(args);

    std::cout << "GEMM Status: " << cutlass::cutlassGetStatusString(status) << std::endl;

    bool acc_passed = learn_util::check_tensor_accuracy(
        M, N, K, alpha, beta,
        A, B, C, C_ref);

    std::cout << "Accuracy check: " << (acc_passed ? "Passed" : "Failed") << std::endl;
}
