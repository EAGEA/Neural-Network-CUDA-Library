//
// Created by Emilien Aufauvre on 29/10/2021.
//

#ifndef CUDANN_MATRIX_H
#define CUDANN_MATRIX_H

#include "lib/util/util.h"

#include <vector_types.h> // To keep .cpp/.h extensions (cuda types).
#include <cstddef>
#include <string>
#include <utility>
#include <algorithm>
#include <initializer_list>

#define DEFAULT_ID "NaN"


namespace cudaNN
{
    /**
     * Matrix representation, allocating memory on both host and
     * device, and implementing computations with CUDA.
     * A matrix of size N*M has N rows (first dimensions)
     * and M columns (second dimension).
     */
    class matrix
    {
        public:

            matrix() = default;
            matrix(const matrix &m);
            matrix(const matrix &m, std::string id);
            matrix(size_t x, size_t y);
            matrix(size_t x, size_t y, std::string id);
            explicit matrix(std::pair<size_t, size_t> dimensions);
            matrix(std::pair<size_t, size_t> dimensions, std::string id);
            matrix(std::initializer_list<float> values, size_t x, size_t y);
            matrix(std::initializer_list<float> values, size_t x, size_t y, std::string id);
            matrix(std::initializer_list<float> values, std::pair<size_t, size_t> dimensions); 
            matrix(std::initializer_list<float> values, std::pair<size_t, size_t> dimensions, std::string id);
            matrix(const float *values, std::pair<size_t, size_t> dimensions); 
            matrix(const float *values, std::pair<size_t, size_t> dimensions, std::string id);
            ~matrix();

            void allocate(const std::pair<size_t, size_t> &dimensions);
            void free();

            void set_id(const std::string &id);

            const std::string &get_id() const;
            float *get_data() const;
            float *get_data();

            /**
             * @return - the number of rows, and columns of the matrix.
             */
            const std::pair<size_t, size_t> &get_dimensions() const;

            /**
             * @return - the number of values in the matrix.
             */
            size_t get_length() const;

            /**
             * @Operators
             * Boolean operators are working only on host memory.
             * Arithmetic operators are working with device memory
             * and copy the results to the host (i.e. the matrix).
             */
            matrix &operator=(const matrix &m);
            matrix &operator+=(const matrix &m);
            matrix operator+(const matrix &m);
            matrix &operator*=(const matrix &m);
            matrix operator*(const matrix &m);
            float &operator[](const int &i);
            const float &operator[](const int &i) const;
            bool operator==(const matrix &m) const;
            bool operator!=(const matrix &m) const;

            /**
             * Enhanced using CUDA.
             * @return - the sum of all the values in "_data".
             */
            float sum() const;

            /**
             * Enhanced using CUDA.
             * @return - the transpose of the matrix.
             */
            matrix transpose() const;

            /**
             * Print the given matrix (host memory).
             * @param m - the matrix concerned.
             */
            static void print(const matrix &m);

        private:

            std::string _id;
            std::pair<size_t, size_t> _dimensions;
            float *_data = nullptr;
    };


    /**
     * CUDA function wrappers for call on host.
     */
    namespace matrix_cuda
    {
        void start_operation(const matrix &m, float **device_data);
        void end_operation(const matrix &m, float **device_data);
        void add(const dim3 &block_dims, const dim3 &thread_dims,
                 const matrix &m1, const matrix &m2);
        void multiply(const dim3 &block_dims, const dim3 &thread_dims,
                      const matrix &m,
                      const matrix &m1, const matrix &m2);
        void sum(const dim3 &block_dims, const dim3 &thread_dims,
                 float *result, const matrix &m);
        void transpose(const dim3 &block_dims, const dim3 &thread_dims,
                       const matrix &result, const matrix &m);
    }
}


#endif //CUDANN_MATRIX_H