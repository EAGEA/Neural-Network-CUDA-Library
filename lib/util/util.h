//
// Created by Emilien Aufauvre on 30/10/2021.
//

#ifndef CUDANN_UTIL_H
#define CUDANN_UTIL_H


#include <cstdlib>
#include <iostream>
#include <string>


namespace util
{
    uint32_t swap_endian(uint32_t val);

    void print_error(std::string location, std::string err);
    void exit_error();

    /**
     * @param nb_rows
     * @param nb_columns
     * @return the CUDA block/thread configuration such that it
     *         covers a grid of size "nb_rows" * "nb_columns".
     */
    std::pair<dim3, dim3> get_cuda_dims(size_t nb_rows, size_t nb_columns)
};


#endif //CUDANN_UTIL_H