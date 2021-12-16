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

            explicit neural_network(std::initializer_list<layer *> layers);
            ~neural_network();

            virtual void fit(dataset &data,
                             const loss_function_t loss_function,
                             const size_t epochs = 1,
                             const size_t batch_size = 1) override;

            virtual matrix predict(const matrix &features) const override;

        private:

            /**
             * Forward propagation/pass; calculate and store intermediate variables.
             * @param features
             * @return
             */
            matrix _forward_propagation(const matrix &features) const;

            /**
             * Backpropagation; calculate and store the gradients of intermediate
             * variables and parameters, using the given loss function.
             * @param predictions
             * @param labels
             * @param loss_function
             */
            void _backward_propagation(const matrix &predictions, 
                                       const matrix &labels,
                                       const loss_function_t loss_function);

            std::vector<layer *> _layers;
    };
}


#endif //CUDANN_NEURAL_NETWORK_H
