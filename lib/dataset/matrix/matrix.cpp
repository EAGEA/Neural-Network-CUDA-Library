//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "matrix.h"


matrix::matrix(std::pair<size_t, size_t> dimensions)
{
    _dimensions = dimensions;
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
        cudaError_t err = cudaMalloc(&_device_data, _dimensions.first * _dimensions.second * sizeof(float));

        if (err == cudaErrorMemoryAllocation)
        {
            // Invalid.
            util::print_error("matrix::allocate", "memory allocation on device failed");
            util::exit_error();
        }

        // Allocate memory on CPU.
        _host_data = new float[_dimensions.first * _dimensions.second];

        _allocated = true;
    }
}

void matrix::free()
{
    if (allocated)
    {
        cudaFree(_device_data);
        delete[] _host_data;
    }
}

const std::pair<size_t, size_t> matrix::get_dimensions() const
{
    return _dimensions;
}

float matrix::get_host_data(const size_t i) const
{
    return _host_data[i];
}

float matrix::get_host_data(const size_t i, const size_t j) const
{
    // TODO check if _dimensions.first.
    return _host_data[i * _dimensions.first + j];
}

float matrix::get_device_data(const size_t i) const
{
    return _device_data[i];
}

float matrix::get_device_data(const size_t i, const size_t j) const
{
    // TODO check if _dimensions.first.
    return _device_data[i * _dimensions.first + j];
}

void matrix::set_host_data(const size_t i, float f)
{
    _host_data[i] = f;
}

void matrix::set_host_data(const size_t i, const size_t j, float f)
{
    // TODO check if _dimensions.first.
    _host_data[i * _dimensions.first + j] = f;
}

void matrix::set_device_data(const size_t i, float f)
{
    _device_data[i] = f;
}

void matrix::set_device_data(const size_t i, const size_t j, float f)
{
    // TODO check if _dimensions.first.
    _device_data[i * _dimensions.first + j] = f;
}

bool matrix::compare_host_data(const matrix &m) const
{
    if (m.get_dimensions() == _dimensions)
    {
        return false;
    }

    for (size_t i; i < _dimensions.first; i ++)
    {
        for (size_t j; j < _dimensions.second; j ++)
        {
            if (m.get_host_data(i, j) != get_host_data(i, j))
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

    for (size_t i; i < _dimensions.first; i ++)
    {
        for (size_t j; j < _dimensions.second; j ++)
        {
            if (m.get_device_data(i, j) != get_device_data(i, j))
            {
                return false;
            }
        }
    }

    return true;
}

bool matrix::operator==(const matrix &m1, const matrix &m2)
{
    return m1.compare_host_data(m2);
}

bool matrix::operator!=(const matrix &m1, const matrix &m2)
{
    return ! m1.compare_host_data(m2);
}

const float& matrix::operator[](const int i) const
{
    return _host_data.get(i);
}

float& matrix::operator[](const int i)
{
    return _host_data.get(i);
}

const float& Matrix::operator[](const int index) const {
    return data_host.get()[index];
}