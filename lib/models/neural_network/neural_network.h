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
     * of 1 row features and 1 row labels.
     */
    class neural_network: public model
    {
        public:

            neural_network(std::initializer_list<layer *> layers);

            void fit(dataset &data,
                     const function &loss_function,
                     size_t epochs = 1,
                     size_t batch_size = 1,
                     float learning_rate = 0.01) override;

            matrix predict(const matrix &features) const override;
            std::vector<matrix> predict(dataset &test) const override;

        private:

            /**
             * Forward propagation/pass; calculate and store intermediate variables.
             * @param features - from a dataset entry.
             * @return - the neural network predictions.
             */
            matrix _feed_forward(const matrix &features) const;

            /**
             * Backpropagation; calculate and store the gradients of intermediate
             * variables and functions, using the given loss function.
             * @param predictions
             * @param labels
             * @param loss_function
             */
            void _backward_propagation(matrix &predictions,
                                       matrix &labels,
                                       const function &loss_function);

            std::vector<layer *> _layers;
    };
}


#endif //CUDANN_NEURAL_NETWORK_H