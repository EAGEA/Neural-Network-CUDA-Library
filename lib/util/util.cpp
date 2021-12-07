//
// Created by Emilien Aufauvre on 30/10/2021.
//

#include "util.h"


uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void print_error(std::string location, std::string err)
{
    std::cerr << "[ERROR] at "<< location << " > " << err << "." << std::endl;
}

void exit_error()
{
    std::exit(EXIT_FAILURE);
}

std::pair<dim3, dim3> get_cuda_dims(size_t nb_rows, size_t nb_columns)
{
    dim3 blocks_per_grid(1, 1);
    dim3 threads_per_block = dim3(nb_rows, nb_columns);

    if (nb_rows * nb_columns > MAX_NB_THREADS)
    {
        blocks_per_grid.x = std::ceilf((float) nb_rows / (float) MAX_NB_THREADS);
        blocks_per_grid.y = std::ceilf((float) nb_columns / (float) MAX_NB_THREADS);
        threads_per_block.x = MAX_NB_THREADS;
        threads_per_block.y = MAX_NB_THREADS;
    }

    return std::pair<dim3, dim3>(blocks_per_grid, threads_per_block);
}