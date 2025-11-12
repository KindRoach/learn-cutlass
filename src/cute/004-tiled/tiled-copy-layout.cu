#include <cute/tensor.hpp>

#include "cute/utils.cuh"

void tiled_copy_layout_example_val_change()
{
    using namespace cute;
    using T = float;

    auto thr_layout_row = Layout<Shape<_32, _8>, Stride<_8, _1>>{}; // row-major
    auto thr_layout_col = Layout<Shape<_32, _8>, Stride<_1, _32>>{}; // col-major
    auto val_layout_4x1 = Layout<Shape<_4, _1>>{};
    auto val_layout_1x4 = Layout<Shape<_1, _4>>{};

    auto thr_row_val_4x1 = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        thr_layout_row,
        val_layout_4x1
    );
    print_with_label("thr_row_val_4x1: ", thr_row_val_4x1);
    print_latex_to_file("thr_row_val_4x1.latex", thr_row_val_4x1);

    auto thr_row_val_1x4 = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        thr_layout_row,
        val_layout_1x4
    );
    print_with_label("thr_row_val_1x4: ", thr_row_val_1x4);
    print_latex_to_file("thr_row_val_1x4.latex", thr_row_val_1x4);

    auto thr_col_val_4x1 = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        thr_layout_col,
        val_layout_4x1
    );
    print_with_label("thr_col_val_4x1: ", thr_col_val_4x1);
    print_latex_to_file("thr_col_val_4x1.latex", thr_col_val_4x1);

    auto thr_col_val_1x4 = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        thr_layout_col,
        val_layout_1x4
    );
    print_with_label("thr_col_val_1x4: ", thr_col_val_1x4);
    print_latex_to_file("thr_col_val_1x4.latex", thr_col_val_1x4);
}

int main()
{
    tiled_copy_layout_example_val_change();
}
