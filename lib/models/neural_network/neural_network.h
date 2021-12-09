//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_NEURAL_NETWORK_H
#define CUDANN_NEURAL_NETWORK_H

#include "lib/models/model.h"
#include "lib/models/neural_network/layers/layer.h"

#include <initializer_list>



/**
 * Model implementation of a neural network.
 */
class neural_network: public model
{
    public:

        neural_network(std::initializer_list<layer> layers);

        virtual void fit(dataset data,
                         loss_function_t loss_function,
                         size_t epochs = 1,
                         size_t batch_size = 1) override;

        virtual matrix predict(matrix features) override;

    private:

        /**
         * Forward propagation/pass; calculate and store intermediate variables.
         * @param features
         * @return
         */
        matrix _forward_propagation(matrix features);

        /**
         * Backpropagation; calculate and store the gradients of intermediate
         * variables and parameters, using the given loss function.
         * @param predictions
         * @param labels
         * @param loss_function
         */
        void _backward_propagation(matrix predictions, matrix labels,
                                   loss_function_t loss_function);

        std::vector<layer> _layers;
};


#endif //CUDANN_NEURAL_NETWORK_H
