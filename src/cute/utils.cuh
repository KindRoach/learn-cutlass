#pragma once

#include <cute/tensor.hpp>

void print_with_label(const char* label, const auto& copy)
{
    using namespace cute;
    print(label);
    print(copy);
    print("\n");
}

void print_layout_with_label(const char* label, const auto& layout)
{
    using namespace cute;
    print(label);
    print_layout(layout);
    print("\n");
}

__host__ __device__
void print_tensor_with_label(const char* label, const auto& tensor)
{
    using namespace cute;
    print(label);
    print_tensor(tensor);
    print("\n");
}

void print_latex_to_file(const std::string& filename, auto& cute_obj)
{
    // Flush any pending stdout before redirecting
    fflush(stdout);
    FILE* new_stdout = nullptr;

#ifdef _WIN32
    // --- Windows version using the safe freopen_s() ---
    freopen_s(&new_stdout, filename.c_str(), "w", stdout);
#else
    // --- Linux / macOS version using standard freopen() ---
    new_stdout = freopen(filename.c_str(), "w", stdout);
#endif

    // Print LaTeX representation to the file
    cute::print_latex(cute_obj);

    // Flush buffers and restore stdout back to console
    fflush(stdout);

#ifdef _WIN32
    // Restore Windows console
    freopen_s(&new_stdout, "CON", "w", stdout);
#else
    // Restore Unix-like terminal
    freopen("/dev/tty", "w", stdout);
#endif
}
