# CMakeLists.txt for base
set(Module "base")

# Include required
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Library and Cpp filse
set(SRCS src/base.cpp src/util.cpp)

# Execute output
add_executable(test_${Module} test/test_base.cpp  ${SRCS})
target_link_libraries(test_${Module} ${OpenCV_LIBS})
