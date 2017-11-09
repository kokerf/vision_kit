# CMakeLists.txt for alignment
set(Module "alignment")

# Include required
include_directories(include ${Base_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
# External Library and Cpp files
set(EXTER_CPP ${Base_SOURCES_DIR}/util.cpp)

# Execute output
add_executable(test_align2D test/test_align2D.cpp src/alignment.cpp ${EXTER_CPP})
add_executable(test_align1D test/test_align1D.cpp src/alignment.cpp ${EXTER_CPP})
target_link_libraries(test_align2D ${OpenCV_LIBS})
target_link_libraries(test_align1D ${OpenCV_LIBS})