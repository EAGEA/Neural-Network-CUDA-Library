//
// Created by Emilien Aufauvre on 29/10/2021.
//

#include "matrix.h"


using namespace cudaNN;


/**
 * Static class member.
 */
const std::string matrix::DEFAULT_ID = "NaN";


matrix::matrix(const size_t x, const size_t y):
        matrix({}, std::pair<size_t, size_t>(x, y), DEFAULT_ID)
{
}

matrix::matrix(const size_t x, const size_t y, std::string id):
        matrix({}, std::pair<size_t, size_t>(x, y), id)
{
}

matrix::matrix(std::pair<size_t, size_t> dimensions):
        matrix({}, dimensions, DEFAULT_ID)
{
}

matrix::matrix(std::pair<size_t, size_t> dimensions, std::string id):
        matrix({}, dimensions, id)
{
}

matrix::matrix(std::initializer_list<float> values, const size_t x, const size_t y):
        matrix(values, std::pair<size_t, size_t>(x, y), DEFAULT_ID)
{
}

matrix::matrix(std::initializer_list<float> values, const size_t x, const size_t y, 
               std::string id):
        matrix(values, std::pair<size_t, size_t>(x, y), id)
{
}

matrix::matrix(std::initializer_list<float> values, std::pair<size_t, size_t> dimensions):
        matrix(values, dimensions, DEFAULT_ID)
{
}

matrix::matrix(std::initializer_list<float> values, std::pair<size_t, size_t> dimensions,
               std::string id):
        _id(id)
{
    _dimensions = dimensions;
    // Allocate memory on GPU.
    matrix_cuda::allocate(_id, _dimensions, &_device_data);
    // Allocate memory on CPU.
    _host_data = new float[_dimensions.first * _dimensions.second];
    // Get the values.
    std::copy(values.begin(), values.end(), _host_data);
}

matrix::~matrix()
{
    // Desallocate on GPU.
    matrix_cuda::free(_id, _device_data);
    // Desallocate on CPU.
    delete[] _host_data;
}

matrix matrix::add(const matrix &m) const
{
    if (m.get_dimensions() != _dimensions)
    {
        // Invalid.
        util::ERROR("matrix::add", 
                    "matrix::_id " + _id + " + " + m.get_id()
                    + " - Invalid @m size; not the same number "
                    + "of rows or/and columns");
        util::ERROR_EXIT();
    }

    // Prepare the output.
    auto output = matrix(_dimensions, "matrix::add");
    auto cuda_dims = util::get_cuda_dims(_dimensions.first, _dimensions.second);
    // Prepare data of operands.
    copy_host_to_device();
    m.copy_host_to_device();
    // Do the computation.
    matrix_cuda::add(cuda_dims.first, cuda_dims.second,
                     output.get_device_data(),
                     _device_data, m.get_device_data(),
                     _dimensions.first, _dimensions.second);
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
                    + " - Invalid @m size; not the same numberi "
                    + "of rows as the number of columns");
        util::ERROR_EXIT();
    }

    // Prepare the output.
    auto nb_rows = _dimensions.first;
    auto nb_columns = m.get_dimensions().second;
    auto output = matrix(nb_rows, nb_columns, "matrix::multiply");
    auto cuda_dims = util::get_cuda_dims(nb_rows, nb_columns);
    // Prepare data of operands.
    copy_host_to_device();
    m.copy_host_to_device();
    // Do the computation.
    matrix_cuda::multiply(cuda_dims.first, cuda_dims.second,
                          output.get_device_data(),
                          _device_data, m.get_device_data(),
                          _dimensions.first, _dimensions.second,
                          m.get_dimensions().first, m.get_dimensions().second);
    // Retrieve data of output.
    output.copy_device_to_host();

    return output;
}

void matrix::set_id(const std::string id)
{
    _id = id;
}

const std::pair<size_t, size_t> matrix::get_dimensions() const
{
    return _dimensions;
}

size_t matrix::get_length() const
{
    return _dimensions.first * _dimensions.second;
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

const std::string matrix::get_id() const
{
    return _id;
}

bool matrix::compare_host_data(const matrix &m) const
{
    if (m.get_dimensions() != _dimensions)
    {
        return false;
    }

    for (size_t i = 0; i < _dimensions.first; i ++)
    {
        for (size_t j = 0; j < _dimensions.second; j ++)
        {
            if (m[i * _dimensions.second + j]
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
    if (m.get_dimensions() != _dimensions)
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

void matrix::copy_host_to_device() const
{
    matrix_cuda::copy_host_to_device(_id, _host_data, _device_data,
                                     _dimensions.first * _dimensions.second * sizeof(float));
}

void matrix::copy_device_to_host() const
{
    matrix_cuda::copy_device_to_host(_id, _host_data, _device_data,
                                     _dimensions.first * _dimensions.second * sizeof(float));
}

matrix &matrix::operator=(const matrix &m)
{
    if (this == &m)
    {
        return *this;
    }

    _dimensions.first = m.get_dimensions().first;
    _dimensions.second = m.get_dimensions().second;
    _id = DEFAULT_ID;
    // Reallocate host memory.
    delete[] _host_data;
    _host_data = new float[_dimensions.first * _dimensions.second];
    // Reallocate device memory.
    matrix_cuda::free(_id, _device_data);
    matrix_cuda::allocate(_id, _dimensions, &_device_data);
    // Copy the values of host.
    std::copy(m.get_host_data(),
              m.get_host_data() + m.get_length() * sizeof(float),
              _host_data);

    return *this;
}

float &matrix::operator[](const int i)
{
    return _host_data[i];
}

const float &matrix::operator[](const int i) const
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

matrix matrix_operators::operator+(const matrix &m1, const matrix &m2)
{
    return m1.add(m2);
}

matrix matrix_operators::operator*(const matrix &m1, const matrix &m2)
{
    return m1.multiply(m2);
}

bool matrix_operators::operator==(const matrix &m1, const matrix &m2)
{
    return m1.compare_host_data(m2);
}

bool matrix_operators::operator!=(const matrix &m1, const matrix &m2)
{
    return ! m1.compare_host_data(m2);
}
