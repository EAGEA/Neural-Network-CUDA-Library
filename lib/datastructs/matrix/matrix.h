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
#include <initializer_list>


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
            matrix(const size_t &x, const size_t &y);
            matrix(const size_t &x, const size_t &y, std::string id);
            explicit matrix(std::pair<size_t, size_t> dimensions);
            matrix(std::pair<size_t, size_t> dimensions, std::string id);
            matrix(std::initializer_list<float> values, const size_t &x, const size_t &y);
            matrix(std::initializer_list<float> values, const size_t &x, const size_t &y,
                   std::string id);
            matrix(std::initializer_list<float> values, std::pair<size_t, size_t> dimensions); 
            matrix(std::initializer_list<float> values, std::pair<size_t, size_t> dimensions, 
                   std::string id);
            matrix(const float *values, std::pair<size_t, size_t> dimensions); 
            matrix(const float *values, std::pair<size_t, size_t> dimensions, std::string id);
            ~matrix();

            void allocate(const std::pair<size_t, size_t> &dimensions);
            void free();

            matrix add(const matrix &m) const;
            matrix multiply(const matrix &m) const; 

            void set_id(const std::string &id);

            /**
             * @return - the number of rows, and columns of the matrix.
             */
            const std::pair<size_t, size_t> &get_dimensions() const;

            /**
             * @return - the number of values in the matrix.
             */
            size_t get_length() const;

            float *get_host_data() const;
            float *get_device_data() const;
            float *get_host_data();
            float *get_device_data();

            const std::string &get_id() const;

            bool compare_host_data(const matrix &m) const;

            void copy_host_to_device() const;
            void copy_device_to_host() const;

            /**
             * Self operators.
             * Working only on host memory.
             */
            matrix &operator+=(const matrix &m);
            matrix &operator+(const matrix &m);
            matrix &operator*=(const matrix &m);
            matrix &operator*(const matrix &m);
            matrix &operator=(const matrix &m);
            float &operator[](const int &i);
            const float &operator[](const int &i) const;

            /**
             * Print the given matrix (host memory).
             * @m - the matrix concerned.
             */
            static void print(const matrix &m);

            static const std::string DEFAULT_ID;

        private:

            std::pair<size_t, size_t> _dimensions;

            float *_host_data = nullptr;
            float *_device_data = nullptr;

            std::string _id;
    };


    /**
     * Operators:
     * Boolean operators are working only on host memory.
     * Arithmetic operators are working with device memory
     * and copy the results to the host.
     */
    namespace matrix_operators
    {
        bool operator==(const matrix &m1, const matrix &m2);
        bool operator!=(const matrix &m1, const matrix &m2);
    }


    /**
     * CUDA function wrappers for call on host.
     */
    namespace matrix_cuda
    {
        void allocate(const std::string &id,
                      const size_t &length,
                      float **device_data);
        void free(const std::string &id, float *&device_data);
        void copy_host_to_device(const std::string &id,
                                 float *host_data, float *device_data, size_t size);
        void copy_device_to_host(const std::string &id,
                                 float *host_data, float *device_data, size_t size);
        void add(const dim3 &block_dims, const dim3 &thread_dims,
                 float *output,
                 float *data1, float *data2,
                 const size_t &nb_rows, const size_t &nb_cols);
        void multiply(const dim3 &block_dims, const dim3 &thread_dims,
                      float *output,
                      const float *data1, const float *data2,
                      const size_t &nb_rows_1, const size_t &nb_cols_1,
                      const size_t &nb_rows_2, const size_t &nb_cols_2);
    }
}


#endif //CUDANN_MATRIX_H