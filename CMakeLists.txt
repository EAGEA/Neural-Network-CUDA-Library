cmake_minimum_required(VERSION 3.20)
project(CudaNN CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_library(CudaNN lib/models/model.h lib/models/neural_network/neural_network.cpp lib/models/neural_network/neural_network.h lib/models/neural_network/layers/layer.h lib/models/neural_network/layers/linear_layer.cpp lib/models/neural_network/layers/linear_layer.h lib/misc/utils/util.cpp lib/misc/utils/util.h lib/dataset/matrix/matrix.cpp lib/dataset/matrix/matrix.h lib/dataset/dataset.cpp lib/dataset/dataset.h lib/tests/main.cpp lib/util/util.cpp lib/util/util.h lib/dataset/element/element.cpp lib/dataset/element/element.h lib/models/neural_network/layers/layer.cpp)

set_target_properties(CudaNN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
