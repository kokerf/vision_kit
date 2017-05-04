# CMakeLists.txt for base
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_base test/test_base.cpp
    src/base.cpp)
target_link_libraries(test_base ${OpenCV_LIBS})