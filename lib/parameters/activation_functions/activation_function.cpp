//
// Created by Emilien Aufauvre on 10/01/2022.
//

#include "activation_function.h"

#include <utility>


using namespace cudaNN;


activation_function::activation_function(std::string id,
                                         activation_functions_cuda::activation_function_t function,
                                         activation_functions_cuda::activation_function_t function_derivative):
        _id(std::move(id)),
        _function(function),
        _function_derivative(function_derivative)
{
}

matrix activation_function::compute(const cudaNN::matrix &inputs) const
{
    auto outputs = matrix(inputs.get_dimensions(),
                          "activation_functions::" + _id + "("
                          + inputs.get_id() + ")");
    auto cuda_dims = util::get_cuda_dims(inputs.get_dimensions());
    // Do the computations on device.
    _function(cuda_dims.first, cuda_dims.second, outputs, inputs);

    return outputs;
}

matrix activation_function::compute_derivative(const cudaNN::matrix &inputs) const
{
    auto outputs = matrix(inputs.get_dimensions(),
                          "activation_functions::" + _id + "_derivative("
                          + inputs.get_id() + ")");
    auto cuda_dims = util::get_cuda_dims(inputs.get_dimensions());
    // Do the computations on device.
    _function_derivative(cuda_dims.first, cuda_dims.second, outputs, inputs);

    return outputs;
}