//
// Created by Emilien Aufauvre on 29/11/2021.
//

#include "activation_functions.h"


using namespace cudaNN;


matrix activation_functions::linear(matrix inputs)
{
    auto outputs = matrix(inputs.get_dimensions());
    auto cuda_dims = util::get_cuda_dims(
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Prepare data of operand.
    inputs.copy_host_to_device();
    // Do the computations.
    activation_functions_cuda::linear(
            cuda_dims.first, cuda_dims.second,
            outputs.get_device_data(),
            inputs.get_device_data(),
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Retrieve data of output.
    outputs.copy_device_to_host();

    return outputs;
}

matrix activation_functions::binary_step(matrix inputs)
{
    auto outputs = matrix(inputs.get_dimensions());
    auto cuda_dims = util::get_cuda_dims(
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Prepare data of operand.
    inputs.copy_host_to_device();
    // Do the computations.
    activation_functions_cuda::binary_step(
            cuda_dims.first, cuda_dims.second,
            outputs.get_device_data(),
            inputs.get_device_data(),
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Retrieve data of output.
    outputs.copy_device_to_host();

    return outputs;
}

matrix activation_functions::sigmoid(matrix inputs)
{
    auto outputs = matrix(inputs.get_dimensions());
    auto cuda_dims = util::get_cuda_dims(
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Prepare data of operand.
    inputs.copy_host_to_device();
    // Do the computations.
    activation_functions_cuda::sigmoid(
            cuda_dims.first, cuda_dims.second,
            outputs.get_device_data(),
            inputs.get_device_data(),
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Retrieve data of output.
    outputs.copy_device_to_host();

    return outputs;
}

matrix activation_functions::relu(matrix inputs)
{
    auto outputs = matrix(inputs.get_dimensions());
    auto cuda_dims = util::get_cuda_dims(
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Prepare data of operand.
    inputs.copy_host_to_device();
    // Do the computations.
    activation_functions_cuda::relu(
            cuda_dims.first, cuda_dims.second,
            outputs.get_device_data(),
            inputs.get_device_data(),
            inputs.get_dimensions().first,
            inputs.get_dimensions().second);
    // Retrieve data of output.
    outputs.copy_device_to_host();

    return outputs;
}
