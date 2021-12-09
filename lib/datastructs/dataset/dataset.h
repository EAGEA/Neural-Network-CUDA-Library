//
// Created by Emilien Aufauvre on 29/10/2021.
//

#ifndef CUDANN_DATASET_H
#define CUDANN_DATASET_H

#include "lib/datastructs/matrix/matrix.h"
#include "lib/datastructs/dataset/element/element.h"
#include "lib/util/util.h"

#include <cstddef>
#include <vector>
#include <utility>
#include <algorithm>


/**
 * Dataset representation.
 */
class dataset
{
    public:

        dataset();

        /**
         * /!\ Note: copy the "elements", such that the current instance owns
         * the array, and does not depend on any external array.
         * @param elements
         */
        dataset(std::vector<element> &elements);

        void add(matrix features, matrix labels);
        void add(element elem);
        void remove(element elem);
        void remove(size_t i);
        element get(size_t i);
        std::vector<element> get_elements();
        size_t size() const;

        /**
         * @return a matrix that contains all the features concatenated.
         */
        matrix get_features();

        /**
         * @param train_size_ratio represent the proportion of the training dataset
         * (between 0.f and 1.f).
         * @return a dataset for training, and another to test,
         * using the "selection sampling" algorithm to save memory in case
         * of big dataset.
         */
        std::pair<dataset, dataset> train_test_split(float train_size_ratio = 0.8f);

        /**
         * @param batch_size
         * @return a random batch of size "batch_size",
         * using the "selection sampling" algorithm to save memory in case
         * of big dataset.
         */
        dataset get_random_batch(size_t batch_size);

        /**
         * @return and load the MNIST dataset.
         */
        static dataset loadMNIST();

    private:

        std::vector<element> _elements;
};


#endif //CUDANN_DATASET_H
