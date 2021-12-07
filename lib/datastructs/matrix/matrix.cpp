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

matrix matrix::add(const matrix &m) const
{
    if (m.get_dimensions() != _dimensions)
    {
        // Invalid.
        util::print_error("matrix::add", "Invalid @m size; not the same number"
                                         " of rows or/and columns");
        util::exit_error();
    }

    matrix output = matrix(_dimensions);
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(_dimensions.first,
                                                          _dimensions.second);

    __kernel_add<<cuda_dims.first, cuda_dims.second>>(
                    output.get_device_data(),
                    _device_data,
                    m.get_device_data());
}

matrix matrix::multiply(const matrix &m) const
{
    if (_dimensions.second != m.get_dimensions().first)
    {
        // Invalid.
        util::print_error("matrix::multiply", "Invalid @m size; not the same number"
                                              " of rows as the number of columns");
        util::exit_error();
    }

    size_t nb_rows = _dimensions.first;
    size_t nb_columns = m.get_dimensions().second;

    matrix output = matrix(nb_rows, nb_columns);
    std::pair<dim3, dim3> cuda_dims = util::get_cuda_dims(nb_rows,
                                                          nb_columns);

    __kernel_multiply<<cuda_dims.first, cuda_dims.second>>(
                    output.get_device_data(),
                    _device_data,
                    m.get_device_data(),
                    _dimensions.first, _dimensions.second,
                    m.get_dimensions().first, m.get_dimensions().second);
}

const std::pair<size_t, size_t> matrix::get_dimensions() const
{
    return _dimensions;
}

float *matrix::get_host_data()
{
    return _host_data;
}

float *matrix::get_device_data()
{
    return _device_data;
}

float matrix::get_device_data(const size_t i, const size_t j) const
{
    return _device_data[i * _dimensions.second + j];
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

    for (size_t i; i < _dimensions.first; i ++)
    {
        for (size_t j; j < _dimensions.second; j ++)
        {
            if (m.get_host_data[i * _dimensions.second + j]
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

    for (size_t i; i < _dimensions.first; i ++)
    {
        for (size_t j; j < _dimensions.second; j ++)
        {
            if (m.get_device_data[i * _dimensions.second + j]
                != _device_data[i * _dimensions.second + j])
            {
                return false;
            }
        }
    }

    return true;
}

matrix matrix::operator+(const matrix &m1, const matrix &m2)
{
    return add(m2);
}

matrix matrix::operator*(const matrix &m1, const matrix &m2)
{
    return multiply(m2)
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

const float& matrix::operator[](const int index) const
{
    return _host_data.get()[index];
}


/**
 * CUDA.
 */


__global__ void __kernel_add(float *output, float *data1, float *data2,
                             size_t nb_rows, size_t nb_cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread index is in the output dimensions.
    if (row < nb_rows && col < nb_cols)
    {
        float sum = 0.f;

        sum += data1[row * nb_cols + col];
        sum += data2[row * nb_cols + col];

        output[row * nb_cols + col] = sum;
    }
}

__global__ void __kernel_multiply(float *output, float *data1, float *data2,
                                  size_t nb_rows_1, size_t nb_cols_1,
                                  size_t nb_rows_2, size_t nb_cols_2)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread index is in the output dimensions.
    if (row < nb_rows_1 && col < nb_cols_2)
    {
        float sum = .0f;

        for (size_t i = 0; i < nb_cols_1; i ++)
        {
            sum += data1[row * nb_cols_1 + i] * data2[i * nb_cols_2 + col];
        }

        output[row * nb_cols_2 + col] = sum;
    }
}