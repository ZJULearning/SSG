set(CMAKE_CXX_STANDARD 11)

add_executable(test_ssg_index test_ssg_index.cpp)
target_link_libraries(test_ssg_index ${PROJECT_NAME})

add_executable(test_ssg_optimized_search test_ssg_optimized_search.cpp)
target_link_libraries(test_ssg_optimized_search ${PROJECT_NAME})

# [ARC-SJ]: ADA-NNS executable compile
add_executable(test_ssg_optimized_search_ADA_NNS test_ssg_optimized_search.cpp)
target_link_libraries(test_ssg_optimized_search_ADA_NNS ${PROJECT_NAME}_ADA_NNS -ltcmalloc)
target_compile_definitions(test_ssg_optimized_search_ADA_NNS PRIVATE ADA_NNS)

## SJ: Executable for DEEP100M dataste
#add_executable(test_ssg_optimized_search_deep100M test_ssg_optimized_search_deep100M.cpp)
#target_link_libraries(test_ssg_optimized_search_deep100M ${PROJECT_NAME})
