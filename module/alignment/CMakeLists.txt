# CMakeLists.txt for alignment
set(Module "alignment")

# Include required
include_directories(include ${Base_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
# External Library and Cpp files
set(EXTER_CPP ${Base_SOURCES_DIR}/base.cpp)

# Execute output
add_executable(test_${Module} test/test_alignment.cpp src/alignment.cpp ${EXTER_CPP})
message("Opencv LIB ${OpenCV_LIBS}")
target_link_libraries(test_${Module} ${OpenCV_LIBS})