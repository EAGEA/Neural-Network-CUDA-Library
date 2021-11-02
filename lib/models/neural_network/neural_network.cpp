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

void neural_network::fit(dataset data, int epochs = 1, size_t batch_size = 1)
{
    for (int i = 1; i <= epochs; i ++)
    {
        for (int j = 1; j <= x.size() / batch; j ++)
        {
            // Get a sample of "batch size" (features + labels).
            dataset batch = data.get_random_batch(batch_size);
            // Train with it:
            for (int k = 0; k < batch_size; k ++)
            {
                // 1. Forward + backward propagation.
                matrix predictions = _forward_propagation(batch.get_features().at(k));
                _backward_propagation(predictions, batch.get_labels().at(k));
                // 2. Updates of parameters.
                _update_weights();
                _update_biases();
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
    matrix predictions = features;

    for (auto layer_: layers)
    {
        predictions = layer_.forward_propagation(predictions);
    }

    return predictions;
}

void neural_network::_backward_propagation(matrix predictions, matrix labels,
                                           float (*loss_function)(, ))
{
    // TODO
    for ()
    {
        layer_.backward_propagation();
    }
}