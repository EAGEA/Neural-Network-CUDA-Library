//
// Created by Emilien Aufauvre on 30/10/2021.
//

#ifndef CUDANN_UTIL_H
#define CUDANN_UTIL_H

#include <cuda_runtime_api.h> // To keep .cpp/.h extensions.
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#define TERM_RESET   "\033[0m"
#define TERM_RED     "\033[31m"


namespace cudaNN
{
    namespace util
    {
        extern const bool _DEBUG;
        extern const bool _ERROR;
        extern const size_t MAX_NB_THREADS;

        void DEBUG(const std::string &location, const std::string &message);
        void ERROR(const std::string &location, const std::string &message);
        void ERROR(const std::string &location, const std::string &message, cudaError_t err);
        void ERROR_EXIT();

        /**
         * @param nb_rows
         * @param nb_columns
         * @return the CUDA block/thread configuration such that it
         *         covers a grid of size "nb_rows" * "nb_columns".
         */
        std::pair<dim3, dim3> get_cuda_dims(size_t nb_rows, size_t nb_columns);

        uint32_t swap_endian(uint32_t val);
    };
}


#endif //CUDANN_UTIL_H
