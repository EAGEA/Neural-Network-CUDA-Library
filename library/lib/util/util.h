//
// Created by Emilien Aufauvre on 30/10/2021.
//

#ifndef CUDANN_UTIL_H
#define CUDANN_UTIL_H

#include "lib/global.h"

#include <cuda_runtime_api.h> // To keep .cpp/.h extensions.
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <ctime>
#include <iostream>
#include <string>
#include <utility>

#define TERM_RESET   "\033[0m"
#define TERM_RED     "\033[31m"
#define TERM_GREEN   "\033[92m"

#define CUDA_CHECK(ans) { util::CUDA_ASSERT((ans), __FILE__, __LINE__); }


namespace cudaNN
{
    namespace util
    {
        void CUDA_ASSERT(cudaError_t code, const char *file, int line, bool abort = true);

        void INFO(const std::string &location, const std::string &message);
        void DEBUG(const std::string &location, const std::string &message);
        void ERROR(const std::string &location, const std::string &message);
        void ERROR(const std::string &location, const std::string &message, cudaError_t err);
        void ERROR_EXIT();

        /**
         * @return - the smallest integral power of two that is not smaller than "n".
         */
        uint64_t ceil2(uint64_t n);

        /**
         * @param dimensions - the pair <nb_rows, nb_cols> to map on a 2D CUDA grid.
         * @return - the CUDA block/thread configuration such that it
         * covers a grid of size "nb_rows" * "nb_cols".
         */
        std::pair<dim3, dim3> get_cuda_2dims(std::pair<size_t, size_t> dimensions);

        /**
         * @param dimensions - the pair <nb_rows, nb_cols> to map on a 1D CUDA grid.
         * @return - the CUDA block/thread configuration such that it
         * covers a grid of size "nb_rows" * "nb_cols".
         */
        std::pair<dim3, dim3> get_cuda_1dims(std::pair<size_t, size_t> dimensions);

        /**
         * Start the record of GPU execution time
         * @param start_event - the starting event.
         * @param end_event - the ending event.
         */
        void GPU_start_record(cudaEvent_t &start_event, cudaEvent_t &end_event);

        /**
         * End the record of GPU execution time
         * @param time_event - the final execution time (ms).
         * @param start_event - the starting event.
         * @param end_event - the ending event.
         */
        void GPU_end_record(float *time_event, cudaEvent_t &start_event, cudaEvent_t &end_event);

        /**
         * Start the record of CPU execution time
         * @param end_event - the start event.
         */
        void CPU_start_record(float *time_event);

        /**
         * End the record of CPU execution time
         * @param end_event - the end event (ms).
         */
        void CPU_end_record(float *time_event);

        /**
         * Record the cost after the forward pass
         * @param cost - cost computed by the loss function.
         */
        void LOSS_record(float cost);
    };
}


#endif //CUDANN_UTIL_H
