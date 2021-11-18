//
// Created by Emilien Aufauvre on 28/10/2021.
//

#include "neural_network.h"


neural_network::neural_network(layer layer_, ...)
{
    va_list va;
    va_start(va, layer_);

    while (layers)
    {
        _layers.push_back(layer_);
        layer_ = va_arg(va, layer);
    }

    va_end(va);
}

void neural_network::fit(dataset data,
                         matrix (*loss_function)(matrix, matrix) = nullptr,
                         size_t epochs = 1,
                         size_t batch_size = 1)
{
    if (loss_function == nullptr)
    {
        // Invalid.
        util::print_error("neural_network::fit", "Invalid @loss_function");
        util::exit_error();
    }

    for (size_t i = 1; i <= epochs; i ++)
    {
        for (size_t j = 1; j <= x.size() / batch; j ++)
        {
            // Get a sample of "batch size" (features + labels).
            auto batch = data.get_random_batch(batch_size);
            // Train with it:
            for (size_t k = 0; k < batch_size; k ++)
            {
                auto element_ = batch.get(k);
                // Forward + backward propagation.
                auto predictions = _forward_propagation(element_.get_features());
                _backward_propagation(predictions, element_.get_labels(), &loss_function);
            }
        }
    }
}

data neural_network::predict(matrix features)
{
    return _forward_propagation(x);
}

matrix neural_network::_forward_propagation(matrix features)
{
    auto predictions = features;

    for (auto layer_: _layers)
    {
        predictions = layer_.forward_propagation(predictions);
    }

    return predictions;
}

void neural_network::_backward_propagation(matrix predictions, matrix labels,
                                           matrix (*loss_function)(matrix, matrix))
{
    auto errors = _loss_function.cost(predictions, labels);

    for (size_t i = _layers.size() - 1; i >= 0; i --)
    {
        auto layer_ = _layers.at(i);
        errors = layer_.backward_propagation(errors, _loss_function);
    }
}