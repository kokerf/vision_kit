set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/opticalflow)

add_executable(optflow test/test_optflow.cpp)
target_link_libraries(optflow ${OpenCV_LIBS})