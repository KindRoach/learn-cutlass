cmake_minimum_required(VERSION 3.25)

project(Learn-CUTLASS LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

include_directories(${CMAKE_SOURCE_DIR}/cutlass/include)

add_executable(main main.cu)
