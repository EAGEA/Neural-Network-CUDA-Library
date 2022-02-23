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
                         const function &loss_function,
                         size_t epochs /*= 1*/,
                         size_t batch_size /*= 1*/,
                         float learning_rate /*= 0.01*/,
                         bool print_loss /*= true*/,
                         size_t delta_loss /*= 100*/)
{
    size_t nb_batches = data.size() / batch_size;
    size_t entries = 0;

    for (size_t i = 1; i <= epochs; i ++)
    {
        util::INFO("neural_network::fit",
                    "Starting epoch " + std::to_string(i) 
                    + " with " + std::to_string(nb_batches) + " batches");

        // For each epoch, execute the training on batches:
        for (size_t j = 1; j <= nb_batches; j ++)
        {
            // Get a sample of "batch size" (features + labels).
            auto batch = data.get_random_batch(batch_size);
            // Train with it.
            for (size_t k = 0; k < batch_size; k ++)
            {
                auto &e = batch.get(k);
                // Forward + backward propagation.
                auto predictions = _feed_forward(e.get_features());
                auto labels = e.get_labels();
                _backward_propagation(predictions, labels, loss_function);
                // Log + save the loss.
                if (print_loss && (entries ++) % delta_loss == 0)
                {
                    auto loss = std::to_string(loss_function.compute(
                            { &predictions, &labels })[0]);
                    util::INFO("neural_network::_backward_propagation",
                               "loss is " + loss);
                    util::add_to_csv(loss, PATH_LOSS_FILE);
                }
            }

            _gradient_descent(batch_size, learning_rate);
        }
    }
}

matrix neural_network::predict(const matrix &features) const
{
    return _feed_forward(features);
}

std::vector<matrix> neural_network::predict(dataset &test) const
{
    auto predictions = std::vector<matrix>();

    for (const auto& e: test.get_entries())
    {
        predictions.push_back(_feed_forward(e.get_features()));
    }

    return predictions;
}

matrix neural_network::_feed_forward(const matrix &features) const
{
    auto predictions = matrix(features, "neural_network::_feed_forward::predictions");

    for (auto l: _layers)
    {
        predictions = l->feed_forward(predictions);
    }

    return predictions;
}

void neural_network::_backward_propagation(matrix &predictions,
                                           matrix &labels,
                                           const function &loss_function)
{
    auto errors = loss_function.compute_derivatives({ &predictions, &labels });

    for (size_t i = _layers.size(); i > 0; i --)
    {
        _layers[i - 1]->backward_propagation(errors,(i == _layers.size() ?
                                                     nullptr : _layers[i]));
    }
}

void neural_network::_gradient_descent(size_t batch_size, float learning_rate)
{
    for (auto l: _layers)
    {
        l->gradient_descent(batch_size, learning_rate);
    }
}

layer *neural_network::get_layer(int i)
{
    return _layers[i];
}

void neural_network::print(const neural_network &n)
{
    std::cout << "---------------------------" << std::endl;
    std::cout << "Neural Network" << std::endl;
    std::cout << "---------------------------" << std::endl;

    for (auto l: n._layers)
    {
        layer::print(*l);
        std::cout << "---------------------------" << std::endl;
    }
}