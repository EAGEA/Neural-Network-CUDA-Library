//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "matrix.h"

#include <utility>


using namespace cudaNN;


matrix::matrix(const matrix &m):
        matrix(m, m.get_id())
{
}

matrix::matrix(const matrix &m, std::string id):
    matrix(m.get_data(), m.get_dimensions(), std::move(id))
{
}

matrix::matrix(const size_t x, const size_t y):
        matrix({}, std::pair<size_t, size_t>(x, y), DEFAULT_ID)
{
}

matrix::matrix(const size_t x, const size_t y, std::string id):
        matrix({}, std::pair<size_t, size_t>(x, y), std::move(id))
{
}

matrix::matrix(std::pair<size_t, size_t> dimensions):
        matrix({}, dimensions, DEFAULT_ID)
{
}

matrix::matrix(std::pair<size_t, size_t> dimensions, std::string id):
        matrix({}, dimensions, std::move(id))
{
}

matrix::matrix(std::initializer_list<float> values, const size_t x, const size_t y):
        matrix(values, std::pair<size_t, size_t>(x, y), DEFAULT_ID)
{
}

matrix::matrix(std::initializer_list<float> values, const size_t x, const size_t y,
               std::string id):
        matrix(values, std::pair<size_t, size_t>(x, y), std::move(id))
{
}

matrix::matrix(std::initializer_list<float> values, std::pair<size_t, size_t> dimensions):
        matrix(values, dimensions, DEFAULT_ID)
{
}

matrix::matrix(std::initializer_list<float> values, std::pair<size_t, size_t> dimensions,
               std::string id)
{
    _id = std::move(id);
    allocate(dimensions);
    std::copy(values.begin(), values.end(), _data);
}

matrix::matrix(const float *values, std::pair<size_t, size_t> dimensions):
        matrix(values, dimensions, DEFAULT_ID)
{
}

matrix::matrix(const float *values, std::pair<size_t, size_t> dimensions, std::string id)
{
    _id = std::move(id);
    allocate(dimensions);
    std::copy(values, values + get_length() * sizeof(float), _data);
}

matrix::~matrix()
{
    //util::DEBUG("matrix::~matrix", "--- " + _id);
    free();
}

void matrix::allocate(const std::pair<size_t, size_t> &dimensions)
{
    _dimensions.first = dimensions.first;
    _dimensions.second = dimensions.second;
    // Allocate the memory with the given dimensions.
    _data = new float[get_length() * sizeof(float)](); // TODO remove * sizeof(float)......
}

void matrix::free()
{
    // If existing, free previous memory.
    delete[] _data;
    _data = nullptr;
}

void matrix::set_id(const std::string &id)
{
    _id = id;
}

const std::pair<size_t, size_t> &matrix::get_dimensions() const
{
    return _dimensions;
}

size_t matrix::get_length() const
{
    return _dimensions.first * _dimensions.second;
}

float *matrix::get_data() const
{
    return _data;
}

float *matrix::get_data()
{
    return _data;
}

const std::string &matrix::get_id() const
{
    return _id;
}

matrix &matrix::operator=(const matrix &m)
{
    if (this == &m)
    {
        return *this;
    }

    free();
    allocate(m.get_dimensions());
    // Copy the values on host memory.
    std::copy(m.get_data(),
              m.get_data() + get_length() * sizeof(float),
              _data);

    return *this;
}

matrix &matrix::operator+=(const matrix &m)
{
    if (_dimensions != m.get_dimensions())
    {
        // Invalid.
        util::ERROR("matrix::operator+=",
                    "matrix::_id " + _id + " + " + m.get_id()
                    + " >> Invalid @m size; not the same number "
                    + "of rows and/or columns");
        util::ERROR_EXIT();
    }

    if (USE_GPU)
    {
        // Do the computation on device.
        auto cuda_dims = util::get_cuda_dims(_dimensions);
        matrix_cuda::add(cuda_dims.first, cuda_dims.second,*this, m);
    }
    else
    {
        // Do the computation on host.
        for (size_t i = 0; i < get_length(); i ++)
        {
            _data[i] += m.get_data()[i];
        }
    }

    return *this;
}

matrix matrix::operator+(const matrix &m)
{
    matrix m_ = matrix(*this, "add(" + _id + ", " + m.get_id() + ")");
    return (m_ += m);
}

matrix &matrix::operator-=(const matrix &m)
{
    if (_dimensions != m.get_dimensions())
    {
        // Invalid.
        util::ERROR("matrix::operator-=",
                    "matrix::_id " + _id + " + " + m.get_id()
                    + " >> Invalid @m size; not the same number "
                    + "of rows and/or columns");
        util::ERROR_EXIT();
    }

    if (USE_GPU)
    {
        // Do the computation on device.
        auto cuda_dims = util::get_cuda_dims(_dimensions);
        matrix_cuda::subtract(cuda_dims.first, cuda_dims.second,*this, m);
    }
    else
    {
        // Do the computation on host.
        for (size_t i = 0; i < get_length(); i ++)
        {
            _data[i] -= m.get_data()[i];
        }
    }

    return *this;
}

matrix matrix::operator-(const matrix &m)
{
    matrix m_ = matrix(*this, "subtract(" + _id + ", " + m.get_id() + ")");
    return (m_ -= m);
}

matrix &matrix::operator*=(const matrix &m)
{
    if (_dimensions.second != m.get_dimensions().first)
    {
        // Invalid.
        util::ERROR("matrix::operator*=",
                    "matrix::_id " + _id + " + " + m.get_id()
                    + " >> Invalid @m size; not the same number "
                    + "of rows as the number of columns");
        std::cout << _id <<  _dimensions.first << " " << _dimensions.second << std::endl;
        std::cout << m.get_id() << m.get_dimensions().first << " " << m.get_dimensions().second << std::endl;
        util::ERROR_EXIT();
    }

    // Prepare multiplication result.
    auto nb_rows = _dimensions.first;
    auto nb_columns = m.get_dimensions().second;
    matrix output = matrix(nb_rows, nb_columns, "matrix::operator*=::helper");

    if (USE_GPU)
    {
        // Do the computation on device.
        auto cuda_dims = util::get_cuda_dims({ nb_rows, nb_columns });
        matrix_cuda::multiply(cuda_dims.first, cuda_dims.second,
                              output,*this, m);
    }
    else
    {
        // Do the computation on host.
        for (size_t i = 0; i < _dimensions.first; i ++)
        {
            for (size_t j = 0; j < m.get_dimensions().second; j ++)
            {
                for (size_t k = 0; k < _dimensions.second; k ++)
                {
                    output.get_data()[i * nb_rows + j] += _data[i * nb_rows + k]
                                                          * m.get_data()[k * m.get_dimensions().first + j];
                }
            }
        }
    }
    // Get the result.
    *this = output;

    return *this;
}

matrix matrix::operator*(const matrix &m)
{
    matrix m_ = matrix(*this, "mult(" + _id + ", " + m.get_id() + ")");
    return (m_ *= m);
}

matrix &matrix::operator*=(float f)
{
    if (USE_GPU)
    {
        // Do the computation on device.
        auto cuda_dims = util::get_cuda_dims(_dimensions);
        matrix_cuda::multiply(cuda_dims.first, cuda_dims.second, *this, f);
    }
    else
    {
        // Do the computation on host.
        for (size_t i = 0; i < get_length(); i ++)
        {
            _data[i] *= f;
        }
    }

    return *this;
}

matrix matrix::operator*(float f)
{
    matrix m_ = matrix(*this, "mult(" + _id + ", float(" + std::to_string(f) + "))");
    return (m_ *= f);
}

float &matrix::operator[](const int &i)
{
    return _data[i];
}

const float &matrix::operator[](const int &i) const
{
    return _data[i];
}

bool matrix::operator==(const matrix &m) const
{
    if (m.get_dimensions() != _dimensions)
    {
        return false;
    }

    for (size_t i = 0; i < get_length(); i ++)
    {
        if (_data[i] != m[i])
        {
            return false;
        }
    }

    return true;
}

bool matrix::operator!=(const matrix &m) const
{
    return ! (*this == m);
}

matrix matrix::hadamard_product(const matrix &v)
{
    if (_dimensions != v.get_dimensions())
    {
        // Invalid.
        util::ERROR("matrix::hadamard_product",
                    "matrix::_id " + _id + " + " + v.get_id()
                    + " >> Invalid @m size; not the same number "
                    + "of rows and/or columns");
        util::ERROR_EXIT();
    }

    matrix m = matrix(*this, "hadamard_product(" + _id + ", " + v.get_id() + ")");

    if (USE_GPU)
    {
        // Do the computation on device.
        auto cuda_dims = util::get_cuda_dims(_dimensions);
        matrix_cuda::hadamard_product(cuda_dims.first, cuda_dims.second,
                                      m, v);
    }
    else
    {
        // Do the computation on host.
        for (size_t i = 0; i < get_length(); i ++)
        {
            m.get_data()[i] = _data[i] * v.get_data()[i];
        }
    }

    return m;
}

float matrix::sum() const
{
    float sum = 0.f;

    if (get_length() > 0)
    {
        if (USE_GPU)
        {
            // Do the computation on device.
            auto cuda_dims = util::get_cuda_dims(_dimensions);
            matrix_cuda::sum(cuda_dims.first, cuda_dims.second,
                             &sum, *this);
        }
        else
        {
            // Do the computation on host.
            for (size_t i = 0; i < get_length(); i ++)
            {
                sum += _data[i];
            }
        }
    }

    return sum;
}

matrix matrix::transpose() const
{
    matrix m = matrix(_dimensions.second, _dimensions.first, "transpose(" + _id + ")");

    if (USE_GPU)
    {
        // Do the computation on device.
        auto cuda_dims = util::get_cuda_dims(_dimensions);
        matrix_cuda::transpose(cuda_dims.first, cuda_dims.second, m, *this);
    }
    else
    {
        // Do the computation on host.
        for (size_t i = 0; i < _dimensions.first; i ++)
        {
            for (size_t j = 0; j < _dimensions.second; j ++)
            {
                m[j * _dimensions.first + i] = _data[i * _dimensions.second + j];
            }
        }
    }

    return m;
}

void matrix::print(const matrix &m)
{
    if (! m.get_id().empty())
    {
        std::cout << "> ID: "
                  << m.get_id()
                  << " <"  
                  << std::endl;
    }

    for (size_t j = 0; j < m.get_dimensions().second; j ++)
    {
        std::cout << "--" << "\t";
    }

    std::cout << std::endl;

    for (size_t i = 0; i < m.get_dimensions().first; i ++)
    {
        std::cout << "|";

        for (size_t j = 0; j < m.get_dimensions().second; j ++)
        {
            std::cout << m[i * m.get_dimensions().second + j] << "\t";
        }

        std::cout << "|" << std::endl;
    }

    for (size_t j = 0; j < m.get_dimensions().second; j ++)
    {
        std::cout << "--" << "\t";
    }

    std::cout << std::endl;
}