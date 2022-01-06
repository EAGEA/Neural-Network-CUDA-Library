//
// Created by Emilien Aufauvre on 29/10/2021.
//

#ifndef CUDANN_DATASET_H
#define CUDANN_DATASET_H

#include "lib/datastructs/matrix/matrix.h"
#include "lib/datastructs/dataset/element/entry.h"
#include "lib/util/util.h"

#include <cstddef>
#include <vector>
#include <array>
#include <utility>
#include <algorithm>
#include <random>


namespace cudaNN
{
    /**
     * Dataset representation.
     */
    class dataset
    {
        public:

            dataset();
            explicit dataset(std::vector<entry *> &entries);
            ~dataset();

            void add(const matrix *features, const matrix *labels);
            void add(entry *e);

            entry &get(size_t i);
            std::vector<entry *> &get_entries();

            size_t size() const;

            /**
             * @return - a matrix that contains all the features concatenated.
             */
            matrix get_features() const;

            /**
             * @return - a matrix that contains all the labels concatenated.
             */
            matrix get_labels() const;

            /**
             * @warning - the split contains pointers from the full dataset, so you
             * have to free/delete either the full dataset or the split, but not both
             * otherwise you will try to delete null pointers.
             * @param train_size_ratio - represent the proportion of the training dataset
             * (between 0.f and 1.f).
             * @return - a partition of the dataset. The first part is for training,
             * and the other for testing.
             */
            std::pair<dataset *, dataset *> train_test_split(float train_size_ratio = 0.8f);

            /**
             * @param batch_size - the size of the batch.
             * @return - a random batch of size "batch_size".
             */
            dataset get_random_batch(size_t batch_size);


            /**
             * @Multiplication_dataset:
             * - for the "MULT_SIZE" entry, generate "MULT_N" random numbers
             * in [0, "MULT_MAX"[ (the features), and associate them to the result
             * of the multiplication between them (the label).
             */
            static const size_t MULT_SIZE = 100;
            static const size_t MULT_NB_FEATURES = 3;
            static const size_t MULT_NB_LABELS = 1;
            static const size_t MULT_MAX = 10;
            static dataset *load_mult();

            /**
             * Print the given dataset.
             * @param d - the dataset concerned.
             */
            static void print(dataset &d);

        private:

            std::vector<entry *> _entries;
    };
}


#endif //CUDANN_DATASET_H
