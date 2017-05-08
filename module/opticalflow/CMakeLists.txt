# CMakeLists.txt for opticalflow
set(Module "opticalflow")

# Include required
include_directories(include ${Base_INCLUDE_DIR})
# Library and Cpp files
set(SRCS src/optical_flow.cpp ${Base_SOURCES_DIR}/base.cpp)

# Execute output
add_executable(test_opticalflow test/test_optical_flow.cpp ${SRCS})
target_link_libraries(test_${Module} ${OpenCV_LIBS})