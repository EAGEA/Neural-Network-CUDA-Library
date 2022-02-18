//
// Created by Emilien Aufauvre on 11/01/2022.
//

#ifndef CUDANN_FUNCTION_H
#define CUDANN_FUNCTION_H

#include "lib/data_structures/matrix/matrix.h"

#include <vector>


namespace cudaNN
{
    typedef void (*function_t)(std::vector<matrix *>);


    /**
     * Abstract wrapper to execute a function or its derivative on matrices.
     * The wrapped functions should be of the form "function_t", such that
     * the first matrix is the output (results), and the following the parameters.
     */
    class function
    {
        public:

            function(std::string id, function_t f, function_t df);

            /**
             * @param inputs - the matrices to be used for computation.
             * @return - the result of the function "_f" on "inputs".
             */
            matrix compute(std::vector<matrix *> inputs) const;

            /**
             * @param inputs - the matrices to be used for computation.
             * @return - the result of the derivative "_df" on "inputs".
             */
            matrix compute_derivatives(std::vector<matrix *> inputs) const;

            std::string get_id() const;

        private:

            const std::string _id;
            const function_t _f;
            const function_t _df;
    };
}


#endif //CUDANN_FUNCTION_H