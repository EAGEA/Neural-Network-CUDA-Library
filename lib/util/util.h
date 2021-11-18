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
};


#endif //CUDANN_UTIL_H