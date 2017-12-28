cmake_minimum_required(VERSION 2.8.3)
project(vision_kit)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
message("Build type: " ${CMAKE_BUILD_TYPE})

# Debug && Output messages
if(CMAKE_BUILD_TYPE MATCHES "Debug")
	add_definitions(-D_DEBUG_MODE_)
endif(CMAKE_BUILD_TYPE MATCHES "Debug")

set(DEBUG_OUTPUT_MESSAGES true)
if(DEBUG_OUTPUT_MESSAGES)
	add_definitions(-D_OUTPUT_MESSAGES_)
endif(DEBUG_OUTPUT_MESSAGES)

if( CMAKE_BUILD_TYPE MATCHES "Debug" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -g -march=native -fopenmp")
else( CMAKE_BUILD_TYPE MATCHES "Release" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -O3 -march=native  -fopenmp")
endif( CMAKE_BUILD_TYPE MATCHES "Debug" )

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# Library required
find_package(OpenCV 3.1.0 EXACT REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if(OPENCV_XFEATURES2D_FOUND)
    add_definitions(-DUSE_XFEATURES2D)
endif(OPENCV_XFEATURES2D_FOUND)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

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

# Module alignment
add_subdirectory(module/alignment)
set(Alignment_INCLUDE_DIR ${PROJECT_SOURCE_DIR}module/alignment/include)