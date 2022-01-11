//
// Created by Emilien Aufauvre on 11/01/2022.
//

#include "function.h"
#include "lib/models/neural_network/functions/activation_functions/activation_functions.h"
#include "lib/models/neural_network/functions/loss_functions/loss_functions.h"


using namespace cudaNN;



function::function(std::string id, function_t f, function_t f_derivative):
        _id(std::move(id)),
        _function(f),
        _function_derivative(f_derivative)
{

}

matrix function::compute(std::vector<matrix *> inputs) const
{
    auto outputs = matrix(inputs[0]->get_dimensions(),
                          "function::" + _id + "("
                          + inputs[0]->get_id() + ")");
    auto cuda_dims = util::get_cuda_dims(inputs[0]->get_dimensions());
    inputs.insert(inputs.begin(), &outputs);
    // Do the computations on device.
    _function(cuda_dims.first, cuda_dims.second, inputs);

    return outputs;
}

matrix function::compute_derivatives(std::vector<matrix *> inputs) const
{
    auto outputs = matrix(inputs[0]->get_dimensions(),
                          "function::" + _id + "_derivative("
                          + inputs[0]->get_id() + ")");
    auto cuda_dims = util::get_cuda_dims(inputs[0]->get_dimensions());
    inputs.insert(inputs.begin(), &outputs);
    // Do the computations on device.
    _function_derivative(cuda_dims.first, cuda_dims.second, inputs);

    return outputs;
}