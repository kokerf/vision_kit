cmake_minimum_required(VERSION 2.8.3)
project(vision_kit)

set(CMAKE_BUILE_TYPE DEBUG)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV 3.1.0 EXACT REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(module/base)
add_subdirectory(module/opticalflow)