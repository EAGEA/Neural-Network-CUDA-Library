cmake_minimum_required(VERSION 3.20)
project(CudaNN CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_library(CudaNN lib/models/model.h lib/models/neural_network/neural_network.cpp lib/models/neural_network/neural_network.h lib/models/neural_network/layers/layer.h lib/models/neural_network/layers/linear_layer.cpp lib/models/neural_network/layers/linear_layer.h lib/parameters/utils/util.cpp lib/parameters/utils/util.h lib/datastructs/matrix/matrix.cpp lib/datastructs/matrix/matrix.h lib/datastructs/dataset/dataset.cpp lib/datastructs/dataset/dataset.h tests/main.cpp lib/util/util.cpp lib/util/util.h lib/datastructs/dataset/element/element.cpp lib/datastructs/dataset/element/element.h lib/models/neural_network/layers/layer.cpp lib/parameters/loss_functions/loss_functions.h lib/parameters/loss_functions/loss_functions.cpp lib/parameters/activation_functions/activation_functions.cpp lib/parameters/activation_functions/activation_functions.h)

set_target_properties(CudaNN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
