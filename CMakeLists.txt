cmake_minimum_required(VERSION 3.25)

project(Learn-CUTLASS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Get a list of all .cpp files in the root directory
set(src_root "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE sources "${src_root}" "${src_root}/*.cu")

# Create an executable target for each .cu file
foreach (file_path ${sources})
    # Extract executable name (from filename without extension)
    get_filename_component(target_name "${file_path}" NAME_WE)

    # Add target
    add_executable("${target_name}" "${file_path}")

    # Include the src directory for headers
    target_include_directories("${target_name}" PRIVATE "${src_root}"
            "${CMAKE_SOURCE_DIR}/cpp-bench-utils/include"
            "${CMAKE_SOURCE_DIR}/cutlass/include"
            "${CMAKE_SOURCE_DIR}/cutlass/tools/util/include"
    )

    if (MSVC)
        target_compile_options("${target_name}" PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>")
    endif ()

    # Set CUDA architectures
    set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES native)

    # Extract the directory of the file
    get_filename_component(target_dir "${file_path}" DIRECTORY)

    # Compute relative path to src root for output placement
    file(RELATIVE_PATH target_dir "${src_root}" "${target_dir}")

    set_target_properties("${target_name}" PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${target_dir}"
    )
endforeach ()

# Copy compile_commands.json if exists
set(cc_file ${CMAKE_BINARY_DIR}/compile_commands.json)
if (EXISTS ${cc_file})
    file(COPY ${cc_file} DESTINATION ${CMAKE_SOURCE_DIR})
endif ()
