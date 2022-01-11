//
// Created by Emilien Aufauvre on 11/01/2022.
//

#ifndef CUDANN_FUNCTION_H
#define CUDANN_FUNCTION_H

#include "lib/data_structures/matrix/matrix.h"

#include <vector>


namespace cudaNN
{
    typedef void (*function_t)(dim3 block_dims, dim3 thread_dims, std::vector<matrix *>);


    class function
    {
        public:

            function(std::string id, function_t f, function_t f_derivative);

            /**
             * @param inputs - the matrices to be used for computation.
             * @return - the result of the function "_function" on "inputs".
             */
            matrix compute(std::vector<matrix *> inputs) const;

            /**
             * @param inputs - the matrices to be used for computation.
             * @return - the result of the derivative
             * "_function_derivative" on "inputs".
             */
            matrix compute_derivative(std::vector<matrix *> inputs) const;

        private:

            const std::string _id;
            const function_t _function;
            const function_t _function_derivative;
    };
}


#endif //CUDANN_FUNCTION_H