set(CMAKE_CXX_STANDARD 11)

# file(GLOB_RECURSE CPP_SOURCES *.cpp)
list(
    APPEND CPP_SOURCES
    index.cpp
    index_random.cpp
    index_ssg.cpp
    util.cpp
)

add_library(${PROJECT_NAME} ${CPP_SOURCES})
add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
#pybind11_add_module(pyssg python.cpp ${CPP_SOURCES})

# [ARC-SJ]: ADA-NNS library
add_library(${PROJECT_NAME}_ADA_NNS STATIC ${CPP_SOURCES})
target_compile_definitions(${PROJECT_NAME}_ADA_NNS PRIVATE ADA_NNS)

install()
