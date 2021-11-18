//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "matrix.h"


matrix::matrix(std::pair<size_t, size_t> dimensions)
{
    // TODO
    _dimensions = dimensions;
}

matrix::matrix(size_t x, size_t y)
{
    matrix(std::pair<size_t, size_t>(x, y));
}

const std::pair<size_t, size_t> matrix::get_dimensions() const
{
    return _dimensions;
}

const float Matrix::operator[](const size_t index)
{
    // TODO
    return;
}

float& Matrix::operator[](const size_t index)
{
    // TODO
    return;
}
