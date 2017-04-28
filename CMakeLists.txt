cmake_minimum_required(VERSION 2.8)
project(vision_kit)

find_package(OpenCV 3.1.0 EXACT REQUIRED)
#include($ENV{OpenCV_DIR}/OpenCVConfig.cmake) 

set(EXECUTABLE_OUTPUT_PATH ${ROJECT_SOURCE_DIR}/bin)

add_subdirectory(module/opticalflow)