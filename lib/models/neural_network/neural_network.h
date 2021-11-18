//
// Created by Emilien Aufauvre on 28/10/2021.
//

#ifndef CUDANN_NEURAL_NETWORK_H
#define CUDANN_NEURAL_NETWORK_H


/**
 * Model implementation of a neural network.
 */
class neural_network: public neural_network
{
    public:

        neural_network(layer layers, ...);

        virtual void fit(dataset data,
                         matrix (*loss_function)(matrix, matrix) = nullptr,
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
                                   matrix (*loss_function)(matrix, matrix))

        const std::vector<layer> _layers;
};


#endif //CUDANN_NEURAL_NETWORK_H