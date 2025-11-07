#include <cute/tensor.hpp>

#include "cute/utils.cuh"

void tiled_copy_layout_example_val_change()
{
    using namespace cute;
    using T = float;

    auto thr_layout = Layout<Shape<_32, _8>>{};
    auto val_layout_4x1 = Layout<Shape<_4, _1>>{};
    auto val_layout_1x4 = Layout<Shape<_1, _4>>{};

    auto val_4x1 = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        thr_layout,
        val_layout_4x1
    );
    print_with_label("val_4x1: ", val_4x1);
    print_latex_to_file("val_4x1.latex", val_4x1);

    auto val_1x4 = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        thr_layout,
        val_layout_1x4
    );
    print_with_label("val_1x4: ", val_1x4);
    print_latex_to_file("val_1x4.latex", val_1x4);
}


void tiled_copy_layout_example_row_col_major()
{
    using namespace cute;
    using T = float;

    auto thr_layout_row = Layout<Shape<_32, _8>, Stride<_8, _1>>{}; // row-major
    auto thr_layout_col = Layout<Shape<_32, _8>, Stride<_1, _32>>{}; // col-major
    auto val_layout_row = Layout<Shape<_4, _1>, Stride<_1, _1>>{}; // row-major
    auto val_layout_col = Layout<Shape<_4, _1>, Stride<_1, _4>>{}; // col-major

    auto thr_row_val_row = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        thr_layout_row,
        val_layout_row
    );
    print_with_label("Thr_row_Val_row: ", thr_row_val_row);
    print_latex_to_file("Thr_row_Val_row.latex", thr_row_val_row);


    auto thr_col_val_row = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        thr_layout_col,
        val_layout_row
    );
    print_with_label("Thr_col_Val_row: ", thr_col_val_row);
    print_latex_to_file("Thr_col_Val_row.latex", thr_col_val_row);

    auto thr_row_val_col = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        thr_layout_row,
        val_layout_col
    );
    print_with_label("Thr_row_Val_col: ", thr_row_val_col);
    print_latex_to_file("Thr_row_Val_col.latex", thr_row_val_col);

    auto thr_col_val_col = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, T>{},
        thr_layout_col,
        val_layout_col
    );
    print_with_label("Thr_col_Val_col: ", thr_col_val_col);
    print_latex_to_file("Thr_col_Val_col.latex", thr_col_val_col);
}

int main()
{
    tiled_copy_layout_example_val_change();
    tiled_copy_layout_example_row_col_major();
}
