//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "matrix.h"

#include <utility>


using namespace cudaNN;


/**
 * Static class member.
 */


matrix::matrix(const matrix &m):
        matrix(m, std::move(m.get_id() + " copy"))
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
    util::DEBUG("matrix::~matrix", "--- " + _id);
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

bool matrix::compare_data(const matrix &m) const
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

matrix &matrix::operator+=(const matrix &m)
{
    if (_dimensions != m.get_dimensions())
    {
        // Invalid.
        util::ERROR("matrix::operator*=",
                    "matrix::_id " + _id + " + " + m.get_id()
                    + " >> Invalid @m size; not the same number "
                    + "of rows and/or columns");
        util::ERROR_EXIT();
    }

    // Do the computation.
    auto cuda_dims = util::get_cuda_dims(_dimensions.first, _dimensions.second);
    matrix_cuda::add(cuda_dims.first, cuda_dims.second,
                     _data, m.get_data(),
                     _dimensions.first, _dimensions.second);

    return *this;
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
        util::ERROR_EXIT();
    }

    // Prepare multiplication result.
    auto nb_rows = _dimensions.first;
    auto nb_columns = m.get_dimensions().second;
    float *output = new float[nb_rows * nb_columns];
    // Do the computation.
    auto cuda_dims = util::get_cuda_dims(_dimensions.first, _dimensions.second);
    matrix_cuda::multiply(cuda_dims.first, cuda_dims.second,
                     output,
                     _data, m.get_data(),
                     _dimensions.first, _dimensions.second,
                     m.get_dimensions().first, m.get_dimensions().second);
    // Get the result.
    free();
    allocate({nb_rows, nb_columns});
    std::copy(output, output + get_length() * sizeof(float), _data);

    return *this;
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

float &matrix::operator[](const int &i)
{
    return _data[i];
}

const float &matrix::operator[](const int &i) const
{
    return _data[i];
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

matrix matrix_operators::operator+(const matrix &m1, const matrix &m2)
{
    matrix m = matrix(m1, "add(" + m1.get_id() + ", " + m2.get_id() + ")");
    return (m += m2);
}

matrix matrix_operators::operator*(const matrix &m1, const matrix &m2)
{
    matrix m = matrix(m1, "mult(" + m1.get_id() + ", " + m2.get_id() + ")");
    return (m *= m2);
}

bool matrix_operators::operator==(const matrix &m1, const matrix &m2)
{
    return m1.compare_data(m2);
}

bool matrix_operators::operator!=(const matrix &m1, const matrix &m2)
{
    return ! m1.compare_data(m2);
}