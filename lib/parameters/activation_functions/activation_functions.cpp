//
// Created by Emilien Aufauvre on 29/11/2021.
//

#include "activation_functions.h"


using namespace cudaNN;


matrix activation_functions::linear(const matrix &inputs)
{
    auto outputs = matrix(inputs.get_dimensions(), 
                          "activation_functions::linear(" 
                          + inputs.get_id() + ")");
    auto cuda_dims = util::get_cuda_dims(
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Do the computations on device.
    activation_functions_cuda::linear(
            cuda_dims.first, cuda_dims.second,
            outputs, inputs);

    return outputs;
}

matrix activation_functions::binary_step(const matrix &inputs)
{
    auto outputs = matrix(inputs.get_dimensions(), 
                          "activation_functions::binary_step("
                          + inputs.get_id() + ")");
    auto cuda_dims = util::get_cuda_dims(
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Do the computations on device.
    activation_functions_cuda::binary_step(
            cuda_dims.first, cuda_dims.second,
            outputs, inputs);

    return outputs;
}

matrix activation_functions::sigmoid(const matrix &inputs)
{
    auto outputs = matrix(inputs.get_dimensions(), 
                          "activation_functions::sigmoid("
                          + inputs.get_id() + ")");
    auto cuda_dims = util::get_cuda_dims(
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Do the computations on device.
    activation_functions_cuda::sigmoid(
            cuda_dims.first, cuda_dims.second,
            outputs, inputs);

    return outputs;
}

matrix activation_functions::relu(const matrix &inputs)
{
    auto outputs = matrix(inputs.get_dimensions(), 
                          "activation_functions::relu("
                          + inputs.get_id() + ")");
    auto cuda_dims = util::get_cuda_dims(
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Do the computations on device.
    activation_functions_cuda::relu(
            cuda_dims.first, cuda_dims.second,
            outputs, inputs);

    return outputs;
}