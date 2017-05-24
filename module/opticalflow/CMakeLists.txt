# CMakeLists.txt for opticalflow
set(Module "opticalflow")

# Include required
include_directories(include ${Base_INCLUDE_DIR})
# External Library and Cpp files
set(EXTER_CPP ${Base_SOURCES_DIR}/base.cpp)

# Execute output
add_executable(test_opticalflow test/test_optical_flow.cpp src/optical_flow.cpp ${EXTER_CPP})
target_link_libraries(test_${Module} ${OpenCV_LIBS})