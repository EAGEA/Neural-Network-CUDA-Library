//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "matrix.h"

#include <utility>


using namespace cudaNN;


/**
 * Static class member.
 */
const std::string matrix::DEFAULT_ID = "NaN";


matrix::matrix(const matrix &m)
{
    _id = m.get_id() + " copy";
    *this = m;
}

matrix::matrix(const size_t &x, const size_t &y):
        matrix({}, std::pair<size_t, size_t>(x, y), DEFAULT_ID)
{
}

matrix::matrix(const size_t &x, const size_t &y, std::string id):
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

matrix::matrix(std::initializer_list<float> values, const size_t &x, const size_t &y):
        matrix(values, std::pair<size_t, size_t>(x, y), DEFAULT_ID)
{
}

matrix::matrix(std::initializer_list<float> values, const size_t &x, const size_t &y,
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
    // Get the values.
    std::copy(values.begin(), values.end(), _host_data);
}

matrix::matrix(const float *values, std::pair<size_t, size_t> dimensions):
        matrix(values, dimensions, DEFAULT_ID)
{
}

matrix::matrix(const float *values, std::pair<size_t, size_t> dimensions, std::string id)
{
    _id = std::move(id);

    allocate(dimensions);
    // Get the values.
    std::copy(values, values + get_length() * sizeof(float),_host_data);
}

matrix::~matrix()
{
    free();
}

void matrix::allocate(const std::pair<size_t, size_t> &dimensions)
{
    _dimensions.first = dimensions.first;
    _dimensions.second = dimensions.second;

    // Allocate the memory with the given dimensions, on both CPU & GPU.
    if (_host_data == nullptr)
    {
        _host_data = new float[get_length()];
    }

    if (_device_data == nullptr)
    {
        matrix_cuda::allocate(_id, get_length(),
                              &_device_data);
    }

    util::DEBUG("matrix::allocate", "+++ " + _id);
}

void matrix::free()
{
    // If existing, free previous memory, both on CPU & GPU.
    if (_host_data != nullptr)
    {
        delete[] _host_data;
        _host_data = nullptr;
    }

    if (_device_data != nullptr)
    {
        matrix_cuda::free(_id, _device_data);
        _device_data = nullptr;
    }

    util::DEBUG("matrix::free", "--- " + _id);
}


matrix matrix::add(const matrix &m) const
{
    if (_dimensions != m.get_dimensions())
    {
        // Invalid.
        util::ERROR("matrix::add", 
                    "matrix::_id " + _id + " + " + m.get_id()
                    + " >> Invalid @m size; not the same number "
                    + "of rows and/or columns");
        util::ERROR_EXIT();
    }

    auto output = matrix(m.get_dimensions(),
                           "add(" + _id + ", " + m.get_id() + ")");
    // Prepare data of operands.
    copy_host_to_device();
    m.copy_host_to_device();
    // Do the computation.
    /*
    auto cuda_dims = util::get_cuda_dims(_dimensions.first, _dimensions.second);
    matrix_cuda::add(cuda_dims.first, cuda_dims.second,
                     output.get_device_data(),
                     _device_data, m.get_device_data(),
                     _dimensions.first, _dimensions.second);
                     */
    // Retrieve data of output.
    output.copy_device_to_host();

    return output;
}

matrix matrix::multiply(const matrix &m) const
{
    if (_dimensions.second != m.get_dimensions().first)
    {
        // Invalid.
        util::ERROR("matrix::multiply", 
                    "matrix::_id " + _id + " + " + m.get_id()
                    + " >> Invalid @m size; not the same number "
                    + "of rows as the number of columns");
        util::ERROR_EXIT();
    }

    auto nb_rows = _dimensions.first;
    auto nb_columns = m.get_dimensions().second;

    auto output = matrix(nb_rows, nb_columns,
                           "multiply(" + _id + ", " + m.get_id() + ")");
    // Prepare data of operands.
    copy_host_to_device();
    m.copy_host_to_device();
    // Do the computation.
    /*
    auto cuda_dims = util::get_cuda_dims(nb_rows, nb_columns);
    matrix_cuda::multiply(cuda_dims.first, cuda_dims.second,
                          output.get_device_data(), 
                          _device_data, m.get_device_data(),
                          _dimensions.first, _dimensions.second,
                          m.get_dimensions().first, m.get_dimensions().second);
                          */
    // Retrieve data of output.
    output.copy_device_to_host();

    return output;
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

float *matrix::get_host_data() const
{
    return _host_data;
}

float *matrix::get_device_data() const
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

const std::string &matrix::get_id() const
{
    return _id;
}

bool matrix::compare_host_data(const matrix &m) const
{
    if (m.get_dimensions() != _dimensions)
    {
        return false;
    }

    for (size_t i = 0; i < get_length(); i ++)
    {
        if (_host_data[i] != m[i])
        {
            return false;
        }
    }

    return true;
}

void matrix::copy_host_to_device() const
{
    matrix_cuda::copy_host_to_device(_id,
                                     _host_data, _device_data,
                                     get_length());
}

void matrix::copy_device_to_host() const
{
    matrix_cuda::copy_device_to_host(_id,
                                     _host_data, _device_data,
                                     get_length());
}

matrix &matrix::operator+(const matrix &m)
{
    matrix res;
    res = *this;
    res += m;
    return res;
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

    /*
    for (size_t i = 0; i < get_length(); i ++)
    {
        _host_data[i] += m.get_host_data()[i];
    }
    */

   // float *d = nullptr;
    //matrix_cuda::allocate("", get_length() * sizeof (float), &d);

    copy_host_to_device();
    m.copy_host_to_device();
    // Do the computation.
    auto cuda_dims = util::get_cuda_dims(_dimensions.first, _dimensions.second);
    matrix_cuda::add(cuda_dims.first, cuda_dims.second,
                     _device_data,
                     _device_data, m.get_device_data(),
                     _dimensions.first, _dimensions.second);
    // Retrieve data of output.
    copy_device_to_host();

    return *this;
}

matrix &matrix::operator*(const matrix &m)
{
    matrix res = *this;
    res *= m;
    return res;
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

    auto nb_rows = _dimensions.first;
    auto nb_columns = m.get_dimensions().second;

    if (_dimensions.second != nb_columns)
    {
        free();
        allocate(std::pair<size_t, size_t>(nb_rows, nb_columns));
    }

    /*
    // Prepare data of operands.
    copy_host_to_device();
    m.copy_host_to_device();
    // Do the computation.
    auto cuda_dims = util::get_cuda_dims(nb_rows, nb_columns);
    matrix_cuda::multiply(cuda_dims.first, cuda_dims.second,
                          output.get_device_data(),
                          _device_data, m.get_device_data(),
                          _dimensions.first, _dimensions.second,
                          m.get_dimensions().first, m.get_dimensions().second);
    // Retrieve data of output.
    output.copy_device_to_host();

     */

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
    std::copy(m.get_host_data(),
              m.get_host_data() + get_length() * sizeof(float),
              _host_data);

    return *this;
}

float &matrix::operator[](const int &i)
{
    return _host_data[i];
}

const float &matrix::operator[](const int &i) const
{
    return _host_data[i];
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

bool matrix_operators::operator==(const matrix &m1, const matrix &m2)
{
    return m1.compare_host_data(m2);
}

bool matrix_operators::operator!=(const matrix &m1, const matrix &m2)
{
    return ! m1.compare_host_data(m2);
}