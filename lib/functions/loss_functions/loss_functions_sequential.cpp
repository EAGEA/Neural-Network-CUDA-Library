//
// Created by Emilien Aufauvre on 13/01/2022.
//

#include <cmath>

#include "loss_functions.h"


using namespace cudaNN;


void loss_functions_sequential::mean_squared_error(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = std::pow(m[2]->get_data()[i] - m[1]->get_data()[i], 2.0f);
    }
}

void loss_functions_sequential::mean_squared_error_derivative(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = -2.f * (m[2]->get_data()[i] - m[1]->get_data()[i]);
    }
}

void loss_functions_sequential::mean_absolute_error(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = std::abs(m[2]->get_data()[i] - m[1]->get_data()[i]);
    }
}

void loss_functions_sequential::mean_absolute_error_derivative(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = m[1]->get_data()[i] > m[2]->get_data()[i] ? +1.f : -1.f;
    }
}

void loss_functions_sequential::mean_bias_error(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = m[2]->get_data()[i] - m[1]->get_data()[i];
    }
}

void loss_functions_sequential::mean_bias_error_derivative(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = -1.f;
    }
}

void loss_functions_sequential::hinge_loss(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = std::fmax(0.f, 1.f - m[2]->get_data()[i] * m[1]->get_data()[i]);
    }
}

void loss_functions_sequential::hinge_loss_derivative(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = m[1]->get_data()[i] > 1.f ? 0.f : -m[2]->get_data()[i] * 1.f; // TODO check
    }
}

void loss_functions_sequential::binary_cross_entropy_loss(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = -(m[2]->get_data()[i] * logf(m[1]->get_data()[i])
                          + (1.f - m[2]->get_data()[i]) * logf(1.f - m[1]->get_data()[i]));
    }
}

void loss_functions_sequential::binary_cross_entropy_loss_derivative(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = -(m[2]->get_data()[i] / m[1]->get_data()[i]
                          - (1.f - m[2]->get_data()[i]) / (1.f - m[1]->get_data()[i]));
    }
}