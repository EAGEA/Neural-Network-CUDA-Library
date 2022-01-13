//
// Created by Emilien Aufauvre on 11/01/2022.
//

#include "function.h"
#include "lib/functions/activation_functions/activation_functions.h"
#include "lib/functions/loss_functions/loss_functions.h"


using namespace cudaNN;


function::function(std::string id, function_t f, function_t df):
        _id(std::move(id)),
        _f(f),
        _df(df)
{
}

matrix function::compute(std::vector<matrix *> inputs) const
{
    auto outputs = matrix(inputs[0]->get_dimensions(),
                          "function::" + _id + "("
                          + inputs[0]->get_id() + ")");
    inputs.insert(inputs.begin(), &outputs);
    _f(inputs);

    return outputs;
}

matrix function::compute_derivatives(std::vector<matrix *> inputs) const
{
    auto outputs = matrix(inputs[0]->get_dimensions(),
                          "function::" + _id + "_derivative("
                          + inputs[0]->get_id() + ")");
    inputs.insert(inputs.begin(), &outputs);
    _df(inputs);

    return outputs;
}