#include <iostream>

#include "cutlass/numeric_types.h"

int main()
{
    cutlass::half_t x = 2.25_hf;
    std::cout << "The " << x << "*" << x << "=" << x * x << std::endl;
}
