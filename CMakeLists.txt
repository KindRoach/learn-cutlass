cmake_minimum_required(VERSION 3.25)

project(Learn-CUTLASS LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)

# Set the CUDA Architectures
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# include cutlass
include_directories("${CMAKE_SOURCE_DIR}/cutlass/include")
include_directories("${CMAKE_SOURCE_DIR}/cutlass/tools/util/include")

# Get a list of all .cpp files in the root directory
set(src_root "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE sources "${src_root}" "${src_root}/*.cu")

# Create an executable target for each .cu file
foreach(file_path ${sources})
    get_filename_component(target_name "${file_path}" NAME_WE)
    add_executable("${target_name}" "${file_path}")
    target_include_directories("${target_name}" PRIVATE src)
endforeach()
