//
// Created by Emilien Aufauvre on 13/01/2022.
//

#include "matrix.h"


using namespace cudaNN;


void matrix_sequential::add(const matrix &m1, const matrix &m2)
{
    for (size_t i = 0; i < m1.get_length(); i ++)
    {
        m1.get_data()[i] += m2.get_data()[i];
    }
}

void matrix_sequential::subtract(const matrix &m1, const matrix &m2)
{
    for (size_t i = 0; i < m1.get_length(); i ++)
    {
        m1.get_data()[i] -= m2.get_data()[i];
    }
}

void matrix_sequential::multiply(const matrix &m,
                                 const matrix &m1, const matrix &m2)
{
    for (size_t i = 0; i < m1.get_dimensions().first; i ++)
    {
        for (size_t j = 0; j < m.get_dimensions().second; j ++)
        {
            for (size_t k = 0; k < m1.get_dimensions().second; k ++)
            {
                m.get_data()[i * m.get_dimensions().first + j] +=
                        m1.get_data()[i * m.get_dimensions().first + k]
                        * m2.get_data()[k * m2.get_dimensions().first + j];
            }
        }
    }
}

void matrix_sequential::multiply(const matrix &m, float f)
{
    for (size_t i = 0; i < m.get_length(); i ++)
    {
        m.get_data()[i] *= f;
    }
}

void matrix_sequential::do_hadamard_product(const matrix &v1, const matrix &v2)
{
    for (size_t i = 0; i < v1.get_length(); i ++)
    {
        v1.get_data()[i] *= v2.get_data()[i];
    }
}

void matrix_sequential::do_sum(float *result, const matrix &m)
{
    for (size_t i = 0; i < m.get_length(); i ++)
    {
        *result += m.get_data()[i];
    }
}

void matrix_sequential::do_transpose(matrix &result, const matrix &m)
{
    for (size_t i = 0; i < m.get_dimensions().first; i ++)
    {
        for (size_t j = 0; j < m.get_dimensions().second; j ++)
        {
            result[j * m.get_dimensions().first + i] = m.get_data()[i * m.get_dimensions().second + j];
        }
    }
}