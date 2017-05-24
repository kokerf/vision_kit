cmake_minimum_required(VERSION 2.8.3)
project(vision_kit)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Library required
find_package(OpenCV 3.1.0 EXACT REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if(OPENCV_XFEATURES2D_FOUND)
    add_definitions(-DUSE_XFEATURES2D)
endif(OPENCV_XFEATURES2D_FOUND)

#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})

# Module base
add_subdirectory(module/base)
set(Base_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/module/base/include)
set(Base_SOURCES_DIR ${PROJECT_SOURCE_DIR}/module/base/src)

# Module opticalflow
add_subdirectory(module/opticalflow)
set(Opticalflow_INCLUDE_DIR ${PROJECT_SOURCE_DIR}module/opticalflow/include)

# Module epipolar_geometry
add_subdirectory(module/epipolar_geometry)
set(Epipolar_Geometry_INCLUDE_DIR ${PROJECT_SOURCE_DIR}module/epipolar_geometry/include)