//
// Created by Emilien Aufauvre on 29/10/2021.
//

#ifndef CUDANN_MATRIX_H
#define CUDANN_MATRIX_H

#include "lib/util/util.h"
#include "/usr/local/cuda/include/vector_types.h"

#include <cstddef>
#include <utility>
#include <initializer_list>


/**
 * Matrix representation, allocating memory on both host and
 * device, and implementing computations with CUDA.
 * A matrix of size N*M has N rows (first dimensions)
 * and M columns (second dimension).
 */
class matrix
{
    public:

        explicit matrix(std::pair<size_t, size_t> dimensions);
        matrix(const size_t x, const size_t y);
        matrix(std::initializer_list<float> values, std::pair<size_t, size_t> dimensions);
        matrix(std::initializer_list<float> values, const size_t x, const size_t y);
        ~matrix();

        matrix add(const matrix &m) const;
        matrix multiply(const matrix &m) const;

        /**
         * @return the number of rows, and columns of the matrix.
         */
        const std::pair<size_t, size_t> get_dimensions() const;

        /**
         * @return the number of values in the matrix.
         */
        size_t get_length() const;

        const float *get_host_data() const;
        const float *get_device_data() const;
        float *get_host_data();
        float *get_device_data();

        bool compare_host_data(const matrix &m) const;
        bool compare_device_data(const matrix &m) const;

        void copy_host_to_device() const; 
        void copy_device_to_host() const; 

        /**
         * Operators.
         * Working only on host memory.
         */
        matrix &operator=(const matrix &m);
        float &operator[](const int i); 
        const float &operator[](const int i) const;
        
        /**
         * Print the given matrix (host memory).
         * @m
         */
        static void print(const matrix &m);

    private:

        std::pair<size_t, size_t> _dimensions;

        float *_host_data;
        float *_device_data;
};


/**
 * Operators.
 * Boolean operators are working only on host memory.
 * Aggregation operators are working only on device memory.
 */
namespace matrix_operators
{
    bool operator==(const matrix &m1, const matrix &m2);
    bool operator!=(const matrix &m1, const matrix &m2);
    matrix operator+(const matrix &m1, const matrix& m2);
    matrix operator*(const matrix &m1, const matrix& m2);
}


/**
 * CUDA function wrappers for call on host.
 */
namespace __matrix
{
    void __allocate(const std::pair<size_t, size_t> dimensions, float **device_data);
    void __free(float *&device_data);
    void __add(const dim3 block_dims, const dim3 thread_dims,
               float *output,
               const float *data1, const float *data2,
               const size_t nb_rows, const size_t nb_cols);
    void __multiply(const dim3 block_dims, const dim3 thread_dims,
                    float *output,
                    const float *data1, const float *data2,
                    const size_t nb_rows_1, const size_t nb_cols_1,
                    const size_t nb_rows_2, const size_t nb_cols_2);
    void __copy_host_to_device(float *host_data, float *device_data, size_t size);
    void __copy_device_to_host(float *host_data, float *device_data, size_t size);
}


#endif //CUDANN_MATRIX_H
