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

#define TERM_RESET "\033[0m"
#define TERM_RED   "\033[31m"

#define MAX_NB_THREADS_BLOCK 1024

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
         * @param dimensions - the pair <nb_rows, nb_cols> to map on a CUDA grid.
         * @return - the CUDA block/thread configuration such that it
         * covers a grid of size "nb_rows" * "nb_cols".
         */
        std::pair<dim3, dim3> get_cuda_dims(std::pair<size_t, size_t> dimensions);

        void GPU_start_record(cudaEvent_t &start_event, cudaEvent_t &end_event);
        void GPU_end_record(float *time_event, cudaEvent_t &start_event, cudaEvent_t &end_event);
        void CPU_start_record(float *time_event);
        void CPU_end_record(float *time_event);
    };
}


#endif //CUDANN_UTIL_H
