#pragma once

#include <tuple>
#include <fstream>

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

namespace learn_util
{

    // Create random tensors A, B, C, C_reference with fixed seed and return as tuple
    template <typename Element, typename Layout>
    std::tuple<
        cutlass::HostTensor<Element, Layout>,
        cutlass::HostTensor<Element, Layout>,
        cutlass::HostTensor<Element, Layout>,
        cutlass::HostTensor<Element, Layout>>
    create_random_tensors(
        int m, int n, int k,
        uint64_t seed,
        Element mean = Element(0),
        Element stddev = Element(1),
        int bits_less_than_one = 0)
    {
        cutlass::HostTensor<Element, Layout> A(cutlass::MatrixCoord(m, k));
        cutlass::HostTensor<Element, Layout> B(cutlass::MatrixCoord(k, n));
        cutlass::HostTensor<Element, Layout> C(cutlass::MatrixCoord(m, n));
        cutlass::HostTensor<Element, Layout> C_reference(cutlass::MatrixCoord(m, n));

        cutlass::reference::device::TensorFillRandomGaussian(
            A.device_view(), seed, mean, stddev, bits_less_than_one);
        cutlass::reference::device::TensorFillRandomGaussian(
            B.device_view(), seed * 2019, mean, stddev, bits_less_than_one);
        cutlass::reference::device::TensorFillRandomGaussian(
            C.device_view(), seed * 1993, mean, stddev, bits_less_than_one);

        // Copy C to C_reference on device
        cutlass::device_memory::copy_device_to_device(
            C_reference.device_data(),
            C.device_data(),
            C.capacity());

        // Sync all tensors to host
        A.sync_host();
        B.sync_host();
        C.sync_host();
        C_reference.sync_host();

        return std::make_tuple(std::move(A), std::move(B), std::move(C), std::move(C_reference));
    }

    // Check accuracy: compute reference GEMM and compare to actual output
    template <
        typename Element,
        typename Layout,
        typename ScalarType = Element>
    bool check_tensor_accuracy(
        int m, int n, int k,
        ScalarType alpha,
        ScalarType beta,
        cutlass::HostTensor<Element, Layout> &A,
        cutlass::HostTensor<Element, Layout> &B,
        cutlass::HostTensor<Element, Layout> &C,
        cutlass::HostTensor<Element, Layout> &C_reference)
    {
        // Compute reference GEMM: C_reference = alpha * A * B + beta * C_reference
        cutlass::reference::host::Gemm<
            Element, Layout,
            Element, Layout,
            Element, Layout,
            ScalarType, ScalarType>
            gemm_ref;

        gemm_ref(
            {m, n, k},
            alpha,
            A.host_ref(),
            B.host_ref(),
            beta,
            C_reference.host_ref());

        // Compare reference to computed results.
        C.sync_host();
        bool passed = cutlass::reference::host::TensorEquals(
            C_reference.host_view(),
            C.host_view());

        if (!passed)
        {
            char const *filename = "errors.csv";
            std::ofstream file(filename);
            file << "\n\nACTUAL =\n"
                 << C.host_view() << std::endl;
            file << "\n\nReference =\n"
                 << C_reference.host_view() << std::endl;
        }

        return passed;
    }
}