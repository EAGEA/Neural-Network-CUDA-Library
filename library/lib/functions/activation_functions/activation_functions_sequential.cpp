//
// Created by Emilien Aufauvre on 13/01/2022.
//

#include "activation_functions.h"


using namespace cudaNN;


void activation_functions_sequential::linear(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = m[1]->get_data()[i];
    }
}

void activation_functions_sequential::linear_derivative(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = 1.f;
    }
}

void activation_functions_sequential::binary_step(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = m[1]->get_data()[i] < 0.f ? 0.f : 1.f;
    }
}

void activation_functions_sequential::binary_step_derivative(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = 0.f;
    }
}

void activation_functions_sequential::sigmoid(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = 1.f / (1.f + expf(-m[1]->get_data()[i]));
    }
}

void activation_functions_sequential::sigmoid_derivative(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        float sigmoid = 1.f / (1.f + expf(-m[1]->get_data()[i]));
        m[0]->get_data()[i] = sigmoid * (1.f - sigmoid);
    }
}

void activation_functions_sequential::relu(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = fmax(0.f, m[1]->get_data()[i]);
    }
}

void activation_functions_sequential::relu_derivative(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = m[1]->get_data()[i] > 0.f ? 1.f : 0.f;
    }
}

void activation_functions_sequential::tanh(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = tanhf(m[1]->get_data()[i]);
    }
}

void activation_functions_sequential::tanh_derivative(std::vector<matrix *> m)
{
    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        float tanh_ = tanhf(m[1]->get_data()[i]);
        m[0]->get_data()[i] = 1.f - tanh_ * tanh_;
    }
}

void activation_functions_sequential::softmax(std::vector<matrix *> m)
{
    float sum = 0.f;

    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        sum += expf(m[1]->get_data()[i]);
    }

    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        m[0]->get_data()[i] = expf(m[1]->get_data()[i]) / sum;
    }
}

void activation_functions_sequential::softmax_derivative(std::vector<matrix *> m)
{
    float sum = m[1]->sum();

    for (size_t i = 0; i < m[0]->get_length(); i ++)
    {
        size_t row = i / m[0]->get_dimensions().second;
        size_t col = i % m[0]->get_dimensions().second;
        float softmax_x = expf(m[1]->get_data()[row]) / sum;
        float softmax_y = expf(m[1]->get_data()[col]) / sum;

        if (row == col)
        {
            m[0]->get_data()[i] = softmax_x * (1 - softmax_x);
        }
        else
        {
            m[0]->get_data()[i] = -softmax_x * softmax_y;
        }
    }
}