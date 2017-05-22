# CMakeLists.txt for opticalflow
set(Module "epipolar_geometry")

# Include required
include_directories(include)

# Execute output
add_executable(test_fundamental test/test_fundamental.cpp src/fundamental.cpp)
target_link_libraries(test_fundamental ${OpenCV_LIBS})