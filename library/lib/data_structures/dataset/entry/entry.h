//
// Created by Emilien Aufauvre on 31/10/2021.
//

#ifndef CUDANN_ENTRY_H
#define CUDANN_ENTRY_H

#include "lib/data_structures/matrix/matrix.h"


namespace cudaNN
{
    /**
     * Entry in a dataset. Contains features (to do predictions on),
     * and annotated labels (to be predicted).
     */
    class entry
    {
        public:

            entry(matrix features, matrix labels);
            ~entry();

            const matrix &get_features() const;
            const matrix &get_labels() const;

            bool compare_features(const matrix &features) const;
            bool compare_labels(const matrix &labels) const;
            bool compare(const entry &e) const;

            bool operator==(const entry &e) const;
            bool operator!=(const entry &e) const;

            /**
             * Print the given entry.
             * @param e - the entry concerned.
             */
            static void print(const entry &e);

        private:

            const matrix _features;
            const matrix _labels;
    };
}


#endif //CUDANN_ENTRY_H
