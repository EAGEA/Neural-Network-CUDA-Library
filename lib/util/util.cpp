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

    if (nb_rows * nb_columns > 512)
    {
        // TODO change
        threads_per_block.x = 512;
        threads_per_block.y = 512;
        blocks_per_grid.x = ceil((double) nb_rows / 512.0);
        blocks_per_grid.y = ceil((double) nb_columns / 512.0);
    }

    return std::pair<dim3, dim3>(blocks_per_grid, threads_per_block);
}