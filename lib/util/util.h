//
// Created by Emilien Aufauvre on 30/10/2021.
//

#ifndef CUDANN_UTIL_H
#define CUDANN_UTIL_H


#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <iostream>
#include <string>


namespace util
{
    bool _DEBUG = true;
    bool _ERROR = true;

    void DEBUG(const std::string location, const char *format, ...);
    void ERROR(const std::string location, const char *format, ...);
    void PRINT(const FILE *stream, const char *format, ...);
    void ERROR_EXIT();

    /**
     * @param nb_rows
     * @param nb_columns
     * @return the CUDA block/thread configuration such that it
     *         covers a grid of size "nb_rows" * "nb_columns".
     */
    std::pair<dim3, dim3> get_cuda_dims(size_t nb_rows, size_t nb_columns)

    uint32_t swap_endian(uint32_t val);
};


#endif //CUDANN_UTIL_H