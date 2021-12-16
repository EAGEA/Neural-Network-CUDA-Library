//
// Created by Emilien Aufauvre on 30/10/2021.
//

#include "util.h"


using namespace cudaNN;


const bool util::_DEBUG = true;
const bool util::_ERROR = true;
const size_t util::MAX_NB_THREADS = 512;


void util::DEBUG(const std::string location, const std::string message) 
{
    if (! util::_DEBUG)
    {
        return;
    }

    std::cout << "[DEBUG] at " + location + " >> " + message << std::endl;
}

void util::ERROR(const std::string location, const std::string message) 
{
    if (! util::_ERROR)
    {
        return;
    }

    std::cerr << "[DEBUG] at " + location + " >> " + message << std::endl;
}

void util::ERROR_EXIT()
{
    std::exit(EXIT_FAILURE);
}

std::pair<dim3, dim3> util::get_cuda_dims(size_t nb_rows, size_t nb_columns)
{
    dim3 blocks_per_grid(1, 1);
    dim3 threads_per_block = dim3(nb_columns, nb_rows);

    if (nb_rows * nb_columns > util::MAX_NB_THREADS)
    {
        blocks_per_grid.x = ceil((float) nb_columns / (float) util::MAX_NB_THREADS);
        blocks_per_grid.y = ceil((float) nb_rows / (float) util::MAX_NB_THREADS);
        threads_per_block.x = util::MAX_NB_THREADS;
        threads_per_block.y = util::MAX_NB_THREADS;
    }

    return std::pair<dim3, dim3>(blocks_per_grid, threads_per_block);
}

uint32_t util::swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}
