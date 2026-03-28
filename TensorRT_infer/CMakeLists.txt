cmake_minimum_required(VERSION 3.16)
project(openocr_trt_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set TensorRT_ROOT to your TensorRT install path if needed.
# Example:
#   cmake -S . -B build -DTensorRT_ROOT=/usr/local/TensorRT
if(NOT TensorRT_ROOT)
  set(TensorRT_ROOT /home/hunheyufu/TensorRT-10.10.0.31)
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TensorRT_ROOT}/include /usr/include /usr/local/include)

find_library(TENSORRT_NVINFER nvinfer
  HINTS ${TensorRT_ROOT}/lib ${TensorRT_ROOT}/lib64 /usr/lib /usr/lib/x86_64-linux-gnu)
find_library(TENSORRT_NVONNXPARSER nvonnxparser
  HINTS ${TensorRT_ROOT}/lib ${TensorRT_ROOT}/lib64 /usr/lib /usr/lib/x86_64-linux-gnu)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER)
  message(FATAL_ERROR "TensorRT not found. Please set -DTensorRT_ROOT=<path>.")
endif()

add_executable(infer
  src/infer.cpp
  src/main.cpp
)

# Define PROJECT_ROOT_DIR macro for accessing data files
add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(infer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(infer PRIVATE
  ${TENSORRT_NVINFER}
  ${TENSORRT_NVONNXPARSER}
  CUDA::cudart
  ${OpenCV_LIBS}
)
