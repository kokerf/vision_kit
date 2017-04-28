cmake_minimum_required(VERSION 2.8.3)
project(vision_kit)

SET(CMAKE_BUILE_TYPE DEBUG) 

message(${EXECUTABLE_OUTPUT_PATH})

find_package(OpenCV 3.1.0 EXACT REQUIRED)
#include($ENV{OpenCV_DIR}/OpenCVConfig.cmake) 

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(module/opticalflow)