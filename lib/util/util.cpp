//
// Created by Emilien Aufauvre on 30/10/2021.
//

#include "util.h"


uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}