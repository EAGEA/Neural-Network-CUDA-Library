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