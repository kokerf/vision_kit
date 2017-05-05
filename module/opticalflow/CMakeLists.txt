# CMakeLists.txt for opticalflow
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_opticalflow test/test_optical_flow.cpp src/optical_flow.cpp)
target_link_libraries(test_opticalflow ${OpenCV_LIBS})