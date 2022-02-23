//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_NEURAL_NETWORK_H
#define CUDANN_NEURAL_NETWORK_H

#include "lib/models/model.h"
#include "lib/models/neural_network/layers/layer.h"
#include "lib/util/util.h"

#include <initializer_list>


namespace cudaNN
{
    /**
     * Model implementation of a neural network.
     * Current implementation can be used with dataset 
     * of 1 row features and 1 row labels (horizontal vectors).
     * Is made of layers.
     */
    class neural_network: public model
    {
        public:

            neural_network(std::initializer_list<layer *> layers);

            void fit(dataset &data,
                     const function &loss_function,
                     size_t epochs = 1,
                     size_t batch_size = 1,
                     float learning_rate = 0.01,
                     size_t print_loss = 100) override;

            matrix predict(const matrix &features) const override;
            std::vector<matrix> predict(dataset &test) const override;
            layer *get_layer(int i);


            /**
             * Print the given network (layers).
             * @param n - the network concerned.
             */
            static void print(const neural_network &n);

        private:

            /**
             * Forward propagation/pass; for a given entry, compute the predictions
             * of the model.
             * @param features - from a dataset entry.
             * @return - the neural network predictions.
             */
            matrix _feed_forward(const matrix &features) const;

            /**
             * Backpropagation; calculate and store the gradients of intermediate
             * variables and functions, using the given loss function, for a given
             * entry.
             * @param predictions - the prediction of the model on this entry.
             * @param labels - the ground truth of the entry.
             * @param loss_function - compute the error between the predictions
             * and labels.
             */
            void _backward_propagation(matrix &predictions,
                                       matrix &labels,
                                       const function &loss_function);

            /**
             * Change the model weights and biases in response to the computed
             * errors during the backpropagation.
             * @param batch_size - the number of entries that the model has
             * processed before executing this function.
             * @param learning_rate - determines how much we have to change
             * the model according to the computed errors.
             */
            void _gradient_descent(size_t batch_size, float learning_rate);

            std::vector<layer *> _layers;
    };
}


#endif //CUDANN_NEURAL_NETWORK_H