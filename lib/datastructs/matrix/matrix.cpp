//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "matrix.h"
#include "lib/util/util.h"
#include "/usr/local/cuda/include/vector_types.h"

#include <cstddef>
#include <utility>


matrix::matrix(const std::pair<size_t, size_t> dimensions):
    _dimensions(dimensions)
{
    _allocated = false;
}

matrix::matrix(size_t x, size_t y)
{
    matrix(std::pair<size_t, size_t>(x, y));
}

void matrix::allocate()
{
    if (! _allocated)
    {
        // Allocate memory on GPU.
        __allocate(_dimensions, _device_data);

        // Allocate memory on CPU.
        _host_data = new float[_dimensions.first * _dimensions.second];

        _allocated = true;
    }
}

void matrix::free()
{
    if (_allocated)
    {
        __free(_device_data);
        delete[] _host_data;
    }
}

matrix matrix::add(const matrix &m) const
{
    if (m.get_dimensions() != _dimensions)
    {
        // Invalid.
        util::ERROR("matrix::add", "Invalid @m size; not the same number"
                " of rows or/and columns");
        util::ERROR_EXIT();
    }

    matrix output = matrix(_dimensions);
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(_dimensions.first,
            _dimensions.second);

    __add(cuda_dims.first, cuda_dims.second,
            output.get_device_data(),
            _device_data, m.get_device_data(),
            _dimensions.first, _dimensions.second);

    return output;
}

matrix matrix::multiply(const matrix &m) const
{
    if (_dimensions.second != m.get_dimensions().first)
    {
        // Invalid.
        util::ERROR("matrix::multiply", "Invalid @m size; not the same number"
                                              " of rows as the number of columns");
        util::ERROR_EXIT();
    }

    size_t nb_rows = _dimensions.first;
    size_t nb_columns = m.get_dimensions().second;

    matrix output = matrix(nb_rows, nb_columns);
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(nb_rows,
            nb_columns);

    __multiply(cuda_dims.first, cuda_dims.second,
            output.get_device_data(),
            _device_data, m.get_device_data(),
            _dimensions.first, _dimensions.second,
            m.get_dimensions().first, m.get_dimensions().second);

    return output;
}

const std::pair<size_t, size_t> matrix::get_dimensions() const
{
    return _dimensions;
}

const float *matrix::get_host_data() const
{
    return _host_data;
}

const float *matrix::get_device_data() const
{
    return _device_data;
}

float *matrix::get_host_data()
{
    return _host_data;
}

float *matrix::get_device_data()
{
    return _device_data;
}

void matrix::set_host_data(const size_t i, float f)
{
    _host_data[i] = f;
}

void matrix::set_host_data(const size_t i, const size_t j, float f)
{
    _host_data[i * _dimensions.second + j] = f;
}

void matrix::set_device_data(const size_t i, float f)
{
    _device_data[i] = f;
}

void matrix::set_device_data(const size_t i, const size_t j, float f)
{
    _device_data[i * _dimensions.second + j] = f;
}

bool matrix::compare_host_data(const matrix &m) const
{
    if (m.get_dimensions() == _dimensions)
    {
        return false;
    }

    for (size_t i = 0; i < _dimensions.first; i ++)
    {
        for (size_t j = 0; j < _dimensions.second; j ++)
        {
            if (m.get_host_data()[i * _dimensions.second + j]
                != _host_data[i * _dimensions.second + j])
            {
                return false;
            }
        }
    }

    return true;
}

bool matrix::compare_device_data(const matrix &m) const
{
    if (m.get_dimensions() == _dimensions)
    {
        return false;
    }

    for (size_t i = 0; i < _dimensions.first; i ++)
    {
        for (size_t j = 0; j < _dimensions.second; j ++)
        {
            if (m.get_device_data()[i * _dimensions.second + j]
                != _device_data[i * _dimensions.second + j])
            {
                return false;
            }
        }
    }

    return true;
}

matrix& matrix::operator=(const matrix &m)
{
    if (this == &m)
    {
        return *this;
    }

    // TODO check if need to call free()
    
    // Re-assign for const members.
    this->~matrix();
    new (this) matrix(m.get_dimensions());

    return *this;
}

matrix operator+(const matrix &m1, const matrix &m2)
{
    return m1.add(m2);
}

matrix operator*(const matrix &m1, const matrix &m2)
{
    return m1.multiply(m2);
}

bool operator==(const matrix &m1, const matrix &m2)
{
    return m1.compare_host_data(m2);
}

bool operator!=(const matrix &m1, const matrix &m2)
{
    return ! m1.compare_host_data(m2);
}
