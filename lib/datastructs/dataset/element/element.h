//
// Created by Emilien Aufauvre on 31/10/2021.
//

#ifndef CUDANN_ELEMENT_H
#define CUDANN_ELEMENT_H

#include "lib/datastructs/matrix/matrix.h"


namespace cudaNN
{
    /**
     * Element in a dataset.
     */
    class element
    {
        public:

            element(const matrix &features, const matrix &labels);

            const matrix &get_features() const;
            const matrix &get_labels() const;

            bool compare_features(const matrix &features) const;
            bool compare_labels(const matrix &labels) const;
            bool compare(const element &e) const;

            element &operator=(element &e);

            /**
             * Print the given element.
             * @e
             */
            static void print(const element &e);

        private:

            const matrix &_features;
            const matrix &_labels;
    };


    namespace element_operators
    {
        bool operator==(const element &e1, const element &e2);
        bool operator!=(const element &e1, const element &e2);
    }
}


#endif //CUDANN_ELEMENT_H
