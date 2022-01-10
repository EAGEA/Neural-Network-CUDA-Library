//
// Created by Emilien Aufauvre on 28/10/2021.
//

#include "neural_network.h"


using namespace cudaNN;


neural_network::neural_network(std::initializer_list<layer *> layers):
        _layers(layers)
{
}

void neural_network::fit(dataset &data,
                         const loss_function_t loss_function,
                         const size_t epochs /*= 1*/,
                         const size_t batch_size /*= 1*/)
{
    if (loss_function == nullptr)
    {
        // Invalid.
        util::ERROR("neural_network::fit", "Invalid @loss_function");
        util::ERROR_EXIT();
    }

    for (size_t i = 1; i <= epochs; i ++)
    {
        size_t nb_batches = data.size() / batch_size;

        util::DEBUG("neural_network::fit", 
                    "Starting epoch " + std::to_string(i) 
                    + " with " + std::to_string(nb_batches) + " batches");

        // For each epoch, execute the training on batches:
        for (size_t j = 1; j <= nb_batches; j ++)
        {
            util::DEBUG("neural_network::fit", 
                        "- Starting batch " + std::to_string(j)); 

            // Get a sample of "batch size" (features + labels).
            auto batch = data.get_random_batch(batch_size);
            // Train with it.
            for (size_t k = 0; k < batch_size; k ++)
            {
                auto e = batch.get(k);
                // Forward + backward propagation.
                auto predictions = _forward_propagation(e.get_features());
                //_backward_propagation(predictions, e->get_labels(), loss_function);
            }
        }
    }
}

matrix neural_network::predict(const matrix &features) const
{
    return _forward_propagation(features);
}

std::vector<matrix> neural_network::predict(dataset &test) const
{
    auto predictions = std::vector<matrix>();

    for (const auto& e: test.get_entries())
    {
        predictions.push_back(_forward_propagation(e.get_features()));
    }

    return predictions;
}

matrix neural_network::_forward_propagation(const matrix &features) const
{
    auto predictions = matrix(features, "neural_network::_forward_propagation::predictions");

    for (auto l: _layers)
    {
        predictions = l->forward_propagation(predictions);
    }

    return predictions;
}

void neural_network::_backward_propagation(const matrix &predictions, 
                                           const matrix &labels,
                                           const loss_function_t loss_function)
{
    // One error per neuron in the output layer.
    auto errors = matrix(1, _layers[_layers.size() - 1]->size());

    /*
    for (size_t i = 0; i < errors.get_dimensions().second; i ++)
    {
        errors[i] = loss_function(predictions, labels);
    }
     */

    for (size_t i = _layers.size() - 1; i >= 0; i --)
    {
        errors = _layers.at(i)->backward_propagation(errors);
    }
}