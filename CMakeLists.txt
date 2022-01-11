cmake_minimum_required(VERSION 3.16)

# Set the project name and version #####################################
project(CudaNN LANGUAGES CXX)
# Set standards ########################################################
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Set base repository ##################################################
include_directories(${PROJECT_SOURCE_DIR})
# Include CUDA Libraries ###############################################
find_package(CUDA)

if (CUDA_FOUND)
    enable_language(CUDA)
    # Build Library ####################################################
    add_library(CudaNN
            "lib/data_structures/dataset/dataset.cpp"
            "lib/data_structures/dataset/entry/entry.cpp"
            "lib/data_structures/matrix/matrix.cpp"
            "lib/data_structures/matrix/matrix.cu"
            "lib/models/neural_network/neural_network.cpp"
            "lib/models/neural_network/layers/layer.cpp"
            "lib/models/neural_network/layers/layer.cu"
            "lib/models/neural_network/functions/function.cpp"
            "lib/models/neural_network/functions/function.h"
            "lib/models/neural_network/functions/activation_functions/activation_functions.cu"
            "lib/models/neural_network/functions/loss_functions/loss_functions.cu"
            "lib/util/util.cpp"
            )
    # Build tests ######################################################
    add_executable(test_matrix tests/matrix.cpp)
    target_link_libraries(test_matrix CudaNN)
    ###
    add_executable(test_activation_functions tests/activation_functions.cpp)
    target_link_libraries(test_activation_functions CudaNN)
    ###
    add_executable(test_loss_functions tests/loss_functions.cpp)
    target_link_libraries(test_loss_functions CudaNN)
    ###
    add_executable(test_neural_network tests/neural_network.cpp)
    target_link_libraries(test_neural_network CudaNN)
    ###
endif ()