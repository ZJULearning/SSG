set(CMAKE_CXX_STANDARD 11)

add_executable(test_ssg_index test_ssg_index.cpp)
target_link_libraries(test_ssg_index ${PROJECT_NAME})

add_executable(test_ssg_optimized_search test_ssg_optimized_search.cpp)
target_link_libraries(test_ssg_optimized_search ${PROJECT_NAME})