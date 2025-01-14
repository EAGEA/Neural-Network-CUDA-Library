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
            "lib/data_structures/matrix/matrix_parallel.cu"
            "lib/data_structures/matrix/matrix_sequential.cpp"
            "lib/models/neural_network/neural_network.cpp"
            "lib/models/neural_network/layers/layer.cpp"
            "lib/functions/function.cpp"
            "lib/functions/activation_functions/activation_functions_parallel.cu"
            "lib/functions/activation_functions/activation_functions_sequential.cpp"
            "lib/functions/loss_functions/loss_functions_parallel.cu"
            "lib/functions/loss_functions/loss_functions_sequential.cpp"
            "lib/util/util.cpp"
            examples/neural_network_2.cpp)
    # Build examples ######################################################
    add_executable(matrix examples/matrix.cpp)
    target_link_libraries(matrix CudaNN)
    ###
    add_executable(activation_functions examples/activation_functions.cpp)
    target_link_libraries(activation_functions CudaNN)
    ###
    add_executable(loss_functions examples/loss_functions.cpp)
    target_link_libraries(loss_functions CudaNN)
    ###
    add_executable(neural_network_1 examples/neural_network_1.cpp)
    target_link_libraries(neural_network_1 CudaNN)
    ###
    add_executable(neural_network_2 examples/neural_network_2.cpp)
    target_link_libraries(neural_network_2 CudaNN)
    ###
    add_executable(op_time_matrices examples/op_time_matrices.cpp)
    target_link_libraries(op_time_matrices CudaNN)
    ###
    add_executable(op_time_functions examples/op_time_functions.cpp)
    target_link_libraries(op_time_functions CudaNN)
    ###
    add_executable(debug_backprop examples/debug_backprop.cpp examples/debug_backprop.cpp)
    target_link_libraries(debug_backprop CudaNN)
    ###
endif ()