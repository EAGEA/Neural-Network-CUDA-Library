cmake_minimum_required(VERSION 3.16)

# Set the project name and version ######################
project(CudaNN LANGUAGES CXX)
#########################################################
#########################################################
# Set standards #########################################
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#########################################################
#########################################################
# Include the whole project #############################
include_directories(${PROJECT_SOURCE_DIR})
#########################################################
#########################################################
# CUDA Libraries ########################################
find_package(CUDA)

if (CUDA_FOUND)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS} )
    add_executable(main tests/main.cpp)
endif ()
#########################################################