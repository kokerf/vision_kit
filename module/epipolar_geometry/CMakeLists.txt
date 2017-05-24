# CMakeLists.txt for epipolar_geometry
set(Module "epipolar_geometry")

# Include required
include_directories(include ${Base_INCLUDE_DIR})
# External Library and Cpp files
set(EXTER_CPP ${Base_SOURCES_DIR}/base.cpp)

# Execute output
add_executable(test_fundamental test/test_fundamental.cpp src/fundamental.cpp ${EXTER_CPP})
target_link_libraries(test_fundamental ${OpenCV_LIBS})
add_executable(test_homography test/test_homography.cpp src/homography.cpp ${EXTER_CPP})
target_link_libraries(test_homography ${OpenCV_LIBS})